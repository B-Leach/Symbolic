import ast
import math
import operator
import shutil
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from matplotlib import animation

app = Flask(__name__)

# Lock to prevent concurrent requests from corrupting shared state
_request_lock = threading.Lock()


########################
### GLOBAL VARIABLES ###
########################

domain = 20
global_step = 0.0625
x_train = np.arange(-domain / 2, domain / 2, global_step)

# State variables
y_actual = []
predict_final = []
current_equation = ""


#############################
### SAFE EQUATION PARSER ###
#############################

# Supported functions for user input
SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan": np.arctan,
    "atan": np.arctan,  # alias
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,  # alias
    "exp": np.exp,
    "abs": np.abs,
}

# Operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class SafeEquationEvaluator(ast.NodeVisitor):
    """Safely evaluate mathematical expressions using AST parsing."""

    def __init__(self, x_value):
        self.x_value = x_value

    def visit_Constant(self, node):
        return node.value

    def visit_Num(self, node):  # Python 3.7 compatibility
        return node.n

    def visit_Name(self, node):
        if node.id.lower() == "x":
            return self.x_value
        elif node.id.lower() == "e":
            return math.e
        elif node.id.lower() == "pi":
            return math.pi
        else:
            raise ValueError(f"Unknown variable: {node.id}. Use 'x' as the variable.")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](operand)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id.lower()
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(
                f"Unknown function: {func_name}. Supported: {', '.join(SAFE_FUNCTIONS.keys())}"
            )

        if len(node.args) != 1:
            raise ValueError(f"Function {func_name} requires exactly 1 argument")

        arg = self.visit(node.args[0])
        return SAFE_FUNCTIONS[func_name](arg)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def parse_equation(equation_str, x_values):
    """
    Safely parse and evaluate a mathematical equation.

    Args:
        equation_str: String like "2*sin(3*x) + cos(x)"
        x_values: numpy array of x values

    Returns:
        numpy array of y values

    Raises:
        ValueError: If equation is invalid or contains unsafe operations
    """
    # Clean up the equation string
    equation_str = equation_str.strip()
    if not equation_str:
        raise ValueError("Equation cannot be empty")

    # Replace common notations
    equation_str = equation_str.replace("^", "**")  # Allow ^ for power

    try:
        tree = ast.parse(equation_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid equation syntax: {e}")

    # Evaluate for each x value
    results = []
    for x in x_values:
        try:
            evaluator = SafeEquationEvaluator(x)
            result = evaluator.visit(tree)
            results.append(float(result))
        except Exception as e:
            raise ValueError(f"Error evaluating equation at x={x}: {e}")

    return np.array(results)


def detect_functions_in_equation(equation_str):
    """Detect which mathematical functions are used in the equation."""
    equation_lower = equation_str.lower()
    used_functions = set()

    # Check for each function
    function_mapping = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "arctan": "arctan",
        "atan": "arctan",
        "sqrt": "sqrt",
        "log": "log",
        "ln": "log",
        "exp": "exp",
        "abs": "abs",
    }

    for func_name, gplearn_name in function_mapping.items():
        if func_name + "(" in equation_lower:
            used_functions.add(gplearn_name)

    # Check for power operator - parse AST to detect Pow nodes
    try:
        # Replace ^ with ** for parsing
        eq_normalized = equation_str.replace("^", "**")
        tree = ast.parse(eq_normalized, mode="eval")

        # Walk the AST to find Pow operations
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                used_functions.add("pow")
                break
    except:
        # Fallback to simple string check
        if "**" in equation_str or "^" in equation_str:
            used_functions.add("pow")

    return used_functions


#############################
### GRAPH INITIALIZATION ###
#############################

actual_color = "#ff9500"  # Orange
predicted_color = "#007aff"  # Blue
live_color = "#34c759"  # Green
score_color = "#ff3b30"  # Red

# Configure FFmpeg for h264 encoding (browser compatible)
_ffmpeg_path = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
mpl.rcParams["animation.ffmpeg_path"] = _ffmpeg_path
mpl.rcParams["animation.codec"] = "h264"

# Use a non-interactive backend
mpl.use("Agg")


def create_figure():
    """Create a fresh figure for each animation to avoid state issues."""
    fig, axs = plt.subplots(4, sharex=True, figsize=(12, 14))  # Increased size
    fig.suptitle(
        "Symbolic Regression Learning Progress",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
    )
    fig.set_facecolor("#0f0f0f")  # Match website dark background

    axs[0].set_title("Target Function", fontsize=11, color="#a0a0a0")
    axs[1].set_title("Best Predicted Function", fontsize=11, color="#a0a0a0")
    axs[2].set_title("Live Prediction", fontsize=11, color="#a0a0a0")
    axs[3].set_title("R² Score (0 = poor, 1 = perfect)", fontsize=11, color="#a0a0a0")

    (line_act,) = axs[0].plot([], [], lw=2.5, color=actual_color)
    (line_pred,) = axs[1].plot([], [], lw=2.5, color=predicted_color)
    (line_live,) = axs[2].plot([], [], lw=2.5, color=live_color)
    (line_score,) = axs[3].plot([], [], lw=2.5, color=score_color)

    lines = [line_act, line_pred, line_live, line_score]

    for idx, ax in enumerate(axs):
        ax.set_xlim(-domain / 2, domain / 2)
        if idx == 3:  # R² score graph
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.2, color="#333333", linestyle="-", linewidth=0.5)
        ax.set_facecolor("#1a1a1a")  # Dark background for plots
        # Style the axes
        ax.spines["bottom"].set_color("#333333")
        ax.spines["top"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["right"].set_color("#333333")
        ax.tick_params(axis="x", colors="#666666")
        ax.tick_params(axis="y", colors="#666666")

    plt.tight_layout()
    return fig, axs, lines


############################
### REGRESSOR PARAMETERS ###
############################

# Genetic operators - carefully tuned for better exploration/exploitation
p_crossover = 0.7  # Reduced slightly to allow more mutation
p_subtree_mutation = 0.1  # Increased for better exploration
p_hoist_mutation = 0.05  # Increased for simplification
p_point_mutation = 0.1  # Increased for fine-tuning
p_point_replace = 0.05  # Keep constant mutation moderate

# Complexity control
parsimony_coefficient = 0.001  # Reduced to allow more complex solutions when needed
init_depth = (2, 6)  # Min and max depth for initial population

# Population and evolution
pop_size = 300  # Increased for better diversity
gen_amt = 30  # Increased for more evolution time
tournament_size = 20  # Increased for stronger selection pressure

# Training efficiency
max_samples = 0.9  # Increased to use more training data
stopping_criteria = 0.01  # Tighter stopping criteria for better accuracy


#######################
### CUSTOM FUNCTIONS ###
#######################


# Custom functions for gplearn
def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


def _protected_log(x):
    return np.log(np.abs(x) + 1e-10)


def _protected_div(x1, x2):
    """Protected division that avoids divide by zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.abs(x2) > 1e-6, x1 / x2, 1.0
        )  # Return 1 instead of 0 for better gradients


def _exp(x):
    with np.errstate(over="ignore"):
        return np.where(x < 100, np.exp(x), np.exp(100))


def _pow(x1, x2):
    """Protected power function with better overflow handling."""
    with np.errstate(over="ignore", invalid="ignore"):
        # Protect against problematic cases and preserve sign when appropriate
        abs_x1 = np.abs(x1)
        # Clamp exponents to prevent overflow
        x2_clamped = np.clip(x2, -5, 5)
        result = np.where(abs_x1 < 50, np.power(abs_x1 + 1e-10, x2_clamped), 1.0)
        return result


# Create gplearn function objects
gplearn_arctan = make_function(function=np.arctan, name="arctan", arity=1)
gplearn_exp = make_function(function=_exp, name="exp", arity=1)
gplearn_sqrt = make_function(function=_protected_sqrt, name="sqrt", arity=1)
gplearn_log = make_function(function=_protected_log, name="log", arity=1)
gplearn_pow = make_function(function=_pow, name="pow", arity=2)
gplearn_div = make_function(function=_protected_div, name="div", arity=2)

# Mapping from detected function names to gplearn objects
GPLEARN_FUNCTIONS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arctan": gplearn_arctan,
    "sqrt": gplearn_sqrt,
    "log": gplearn_log,
    "exp": gplearn_exp,
    "abs": "abs",
    "pow": gplearn_pow,
    "div": gplearn_div,
}


#######################
### SCORING METRICS ###
#######################


def _mae(y, y_pred, w=None):
    return np.average(np.abs(y_pred - y), weights=w)


mae_no_wrap = make_fitness(function=_mae, greater_is_better=False, wrap=False)


def calc_r2(y_actual, y_pred):
    """Calculate R² (coefficient of determination)."""
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    r2 = 1 - (ss_res / ss_tot)
    return max(0.0, min(1.0, r2))


def calc_mae(y_actual, y_pred):
    """Calculate Mean Absolute Error."""
    return np.average(np.abs(np.array(y_pred) - np.array(y_actual)))


def create_regressor(function_set):
    """Create a new SymbolicRegressor with the specified function set."""
    return SymbolicRegressor(
        population_size=pop_size,
        tournament_size=tournament_size,
        generations=gen_amt,
        stopping_criteria=stopping_criteria,
        p_crossover=p_crossover,
        p_subtree_mutation=p_subtree_mutation,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
        p_point_replace=p_point_replace,
        metric=mae_no_wrap,
        const_range=(-10.0, 10.0),  # Wider range for constants
        random_state=None,  # Allow randomness for variety
        parsimony_coefficient=parsimony_coefficient,
        function_set=function_set,
        low_memory=False,
        feature_names=["x"],
        max_samples=max_samples,
        n_jobs=-1,
        init_method="half and half",
        init_depth=init_depth,  # Control initial tree depth (min, max)
        verbose=0,  # Reduce logging overhead
        warm_start=False,
    )


##################
### APP ROUTES ###
##################

DEFAULT_EQUATION = "2*sin(x) + cos(2*x)"


@app.route("/")
def index_blank():
    with _request_lock:
        # Create fresh figure
        fig, axs, lines = create_figure()

        def animate_blank(i):
            return lines

        anim = animation.FuncAnimation(
            fig, animate_blank, frames=1, interval=40, blit=True, repeat=False
        )

        full_plot = anim.to_html5_video()
        plt.close(fig)

        return render_template(
            "./index.html",
            equation=DEFAULT_EQUATION,
            ps=pop_size,
            ga=gen_amt,
            r2_score="N/A",
            mae_score="N/A",
            pr_eq_formatted="No prediction made yet...",
            full_plot=full_plot,
            samples=max_samples * 100,
        )


@app.route("/validate", methods=["POST"])
def validate():
    """Validate an equation without training."""
    try:
        data = request.get_json()
        equation_str = data.get("equation", "").strip()

        if not equation_str:
            return jsonify({"valid": False, "error": "Please enter an equation"}), 400

        # Try to parse the equation
        try:
            y_test = parse_equation(
                equation_str, x_train[:10]
            )  # Test with small subset
        except ValueError as e:
            return jsonify({"valid": False, "error": str(e)}), 400

        # Check for NaN or Inf values
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
            return jsonify(
                {
                    "valid": False,
                    "error": "Equation produces invalid values (NaN or Infinity)",
                }
            ), 400

        return jsonify({"valid": True, "message": "Equation is valid"})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


@app.route("/train", methods=["POST"])
def train():
    global y_actual, predict_final, current_equation

    with _request_lock:
        try:
            data = request.get_json()
            equation_str = data.get("equation", "").strip()

            if not equation_str:
                return jsonify(
                    {"success": False, "error": "Please enter an equation"}
                ), 400

            current_equation = equation_str

            # Parse and evaluate the equation
            try:
                y_actual = parse_equation(equation_str, x_train)
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400

            # Check for NaN or Inf values
            if np.any(np.isnan(y_actual)) or np.any(np.isinf(y_actual)):
                return jsonify(
                    {
                        "success": False,
                        "error": "Equation produces invalid values (NaN or Infinity) in the range x = [-10, 10]",
                    }
                ), 400

            # Detect which functions are used and build function set
            used_functions = detect_functions_in_equation(equation_str)

            # Always include basic operations (using our protected div)
            function_set = ["add", "sub", "mul", gplearn_div]

            # Add detected functions
            for func_name in used_functions:
                if func_name in GPLEARN_FUNCTIONS:
                    func = GPLEARN_FUNCTIONS[func_name]
                    if func not in function_set:
                        function_set.append(func)

            # Create regressor with appropriate function set
            sr = create_regressor(tuple(function_set))

            # Train the model
            sr.fit(x_train.reshape(-1, 1), y_actual)

            # Check if training succeeded
            if sr._program is None:
                return jsonify(
                    {
                        "success": False,
                        "error": "Training failed to converge. Try a simpler equation or different parameters.",
                    }
                ), 400

            predict_final = sr.predict(x_train.reshape(-1, 1))
            predicted_equation = format_readable_eq(sr._program)

            # Create animation
            fig, axs, lines = create_figure()

            # Adjust y-axis based on data range
            y_min, y_max = np.min(y_actual), np.max(y_actual)
            y_range = y_max - y_min
            y_padding = max(y_range * 0.1, 1)
            for ax in axs[:3]:
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Animation state
            live_xs_local = []
            live_ys_local = []
            score_ys_local = []

            num_frames = 50

            def animate_all(i):
                if i == 0:
                    return lines

                train_index = round(i / num_frames * len(x_train))
                train_index_begin = round((i - 1) / num_frames * len(x_train))

                xs = x_train[:train_index]
                lines[0].set_data(xs, y_actual[: len(xs)])

                if len(sr._programs) == 0:
                    if sr._program is not None:
                        ys = sr._program.execute(xs.reshape(-1, 1))
                        lines[1].set_data(xs, ys)
                    else:
                        lines[1].set_data(xs, np.zeros_like(xs))
                else:
                    idx = round((i * (len(sr._programs) - 1)) / num_frames)
                    idx = min(idx, len(sr._programs) - 1)

                    if idx >= len(sr._programs) - 1:
                        ys = predict_final
                    else:
                        fitness_set = [
                            ((prg and abs(prg.fitness_)) or 100)
                            for prg in sr._programs[idx]
                        ]
                        best_prog_idx = fitness_set.index(min(fitness_set))
                        prog = sr._programs[idx][best_prog_idx]
                        if prog is not None:
                            ys = prog.execute(x_train.reshape(-1, 1))
                        else:
                            ys = predict_final
                    lines[1].set_data(x_train, ys)

                    xs_slice = x_train[train_index_begin:train_index]
                    if len(xs_slice) > 0:
                        if idx >= len(sr._programs) - 1:
                            ys_slice = predict_final[train_index_begin:train_index]
                        else:
                            ys_slice = ys[train_index_begin:train_index]
                        live_xs_local.extend(xs_slice)
                        live_ys_local.extend(ys_slice)
                    lines[2].set_data(live_xs_local, live_ys_local)

                    r2 = calc_r2(y_actual, ys)
                    score_ys_local.extend(np.full(max(1, len(xs_slice)), r2))
                    if len(live_xs_local) == len(score_ys_local):
                        lines[3].set_data(live_xs_local, score_ys_local)

                return lines

            anim = animation.FuncAnimation(
                fig,
                animate_all,
                frames=num_frames,
                interval=40,
                blit=True,
                repeat=False,
            )

            full_plot = anim.to_html5_video()
            plt.close(fig)

            r2_score = round(calc_r2(y_actual, predict_final), 4)
            mae_score = round(calc_mae(y_actual, predict_final), 4)

            return jsonify(
                {
                    "success": True,
                    "video_html": full_plot,
                    "r2_score": r2_score,
                    "mae_score": mae_score,
                    "pr_eq_formatted": predicted_equation,
                    "target_equation": equation_str,
                }
            )
        except Exception as e:
            import traceback

            return jsonify(
                {"success": False, "error": str(e), "traceback": traceback.format_exc()}
            ), 500


#########################
### HELPER FUNCTIONS ###
#########################


def format_readable_eq(eq):
    """Convert gplearn equation format to readable infix notation."""
    eq = str(eq)
    operation = ""

    for idx, char in enumerate(eq):
        operation += char

        if char == "(":
            valid_op = False
            op = ""
            match operation[:-1]:
                case "add":
                    op = " + "
                    valid_op = True
                case "mul":
                    op = " * "
                    valid_op = True
                case "sub":
                    op = " - "
                    valid_op = True
                case "div":
                    op = " / "
                    valid_op = True
                case "pow":
                    op = " ** "
                    valid_op = True

            if not valid_op:
                return operation + format_readable_eq(eq[idx + 1 : -1]) + ")"

            parenthesis = 0
            first_exp = ""
            second_exp = ""
            assign_to_first = True

            for char_inner in eq[idx + 1 :]:
                if char_inner == " ":
                    continue
                elif char_inner == "(":
                    parenthesis += 1
                elif char_inner == ")":
                    if parenthesis == 0:
                        break
                    parenthesis -= 1
                elif parenthesis == 0 and char_inner == ",":
                    assign_to_first = False
                    continue

                if assign_to_first:
                    first_exp += char_inner
                else:
                    second_exp += char_inner

            return format_readable_eq(first_exp) + op + format_readable_eq(second_exp)

    return operation
