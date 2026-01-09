import math
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


def eq(x):
    """Target equation: e1*cos(c1*x) + e2*sin(c2*x)*e3*arctan(c3*x)"""
    return math.cos(c1 * x) * e1 + math.sin(c2 * x) * e2 * math.atan(c3 * x) * e3


########################
### GLOBAL VARIABLES ###
########################

domain = 20
global_step = 0.0625

c1 = 0.0
e1 = 0.0
c2 = 0.0
e2 = 0.0
c3 = 0.0
e3 = 0.0

low_memory = False

x_train = np.arange(-domain / 2, domain / 2, global_step)
y_actual = []

predict_final = []
live_xs = []
live_ys = []
score_ys = []


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
    fig, axs = plt.subplots(4, sharex=True, figsize=(10, 8))
    fig.suptitle(
        "Symbolic Regression Learning Progress", fontsize=14, fontweight="bold"
    )
    fig.set_facecolor("#f5f5f7")

    axs[0].set_title("Target Function", fontsize=10)
    axs[1].set_title("Best Predicted Function", fontsize=10)
    axs[2].set_title("Live Prediction", fontsize=10)
    axs[3].set_title("R² Score (0 = poor, 1 = perfect)", fontsize=10)

    (line_act,) = axs[0].plot([], [], lw=2, color=actual_color)
    (line_pred,) = axs[1].plot([], [], lw=2, color=predicted_color)
    (line_live,) = axs[2].plot([], [], lw=2, color=live_color)
    (line_score,) = axs[3].plot([], [], lw=2, color=score_color)

    lines = [line_act, line_pred, line_live, line_score]

    for idx, ax in enumerate(axs):
        ax.set_xlim(-domain / 2, domain / 2)
        if idx == 3:  # R² score graph
            ax.set_ylim(-0.1, 1.1)  # R² ranges from 0 to 1 (with small padding)
        else:
            ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#ffffff")

    plt.tight_layout()
    return fig, axs, lines


############################
### REGRESSOR PARAMETERS ###
############################

p_crossover = 0.8
p_subtree_mutation = 0.03
p_hoist_mutation = 0.03
p_point_mutation = 0.02
p_point_replace = 0.05
parsimony_coefficient = 0.005
max_samples = 0.6
pop_size = 150
gen_amt = 20
tournament_size = 10


#######################
### SCORING METRICS ###
#######################

arctan = make_function(function=np.arctan, name="arctan", arity=1)


def _mae(y, y_pred, w=None):
    return np.average(np.abs(y_pred - y), weights=w)


mae_no_wrap = make_fitness(function=_mae, greater_is_better=False, wrap=False)


def calc_r2(y_actual, y_pred):
    """Calculate R² (coefficient of determination).
    Returns value between -inf and 1.0, where 1.0 is perfect fit.
    """
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    r2 = 1 - (ss_res / ss_tot)
    # Clamp to 0-1 range for display (negative R² means worse than baseline)
    return max(0.0, min(1.0, r2))


def calc_mae(y_actual, y_pred):
    """Calculate Mean Absolute Error."""
    return np.average(np.abs(np.array(y_pred) - np.array(y_actual)))


# Create the symbolic regressor
sr = SymbolicRegressor(
    population_size=pop_size,
    tournament_size=tournament_size,
    generations=gen_amt,
    stopping_criteria=0.05,
    p_crossover=p_crossover,
    p_subtree_mutation=p_subtree_mutation,
    p_hoist_mutation=p_hoist_mutation,
    p_point_mutation=p_point_mutation,
    p_point_replace=p_point_replace,
    metric=mae_no_wrap,
    const_range=(-5.0, 5.0),
    random_state=1,
    parsimony_coefficient=parsimony_coefficient,
    function_set=("add", "mul", "sin", "cos", arctan),
    low_memory=low_memory,
    feature_names=["X"],
    max_samples=max_samples,
    n_jobs=-1,
    init_method="half and half",
)


##################
### APP ROUTES ###
##################


@app.route("/")
def index_blank():
    global c1, e1, c2, e2, c3, e3, low_memory
    with _request_lock:
        predicted_equation = "No prediction made yet..."

        # Create fresh figure
        fig, axs, lines = create_figure()

        def animate_blank(i):
            return lines

        anim = animation.FuncAnimation(
            fig, animate_blank, frames=1, interval=40, blit=True, repeat=False
        )

        # Use FFMpegWriter with h264 codec for browser compatibility
        writer = animation.FFMpegWriter(
            fps=25, codec="h264", extra_args=["-pix_fmt", "yuv420p"]
        )
        full_plot = anim.to_html5_video()
        plt.close(fig)

        return render_template(
            "./index.html",
            c1=c1,
            e1=e1,
            c2=c2,
            e2=e2,
            c3=c3,
            e3=e3,
            low_mem=low_memory,
            ps=pop_size,
            ga=gen_amt,
            r2_score="N/A",
            mae_score="N/A",
            pr_eq_formatted=predicted_equation,
            full_plot=full_plot,
            samples=max_samples * 100,
        )


@app.route("/train", methods=["POST"])
def train():
    global c1, e1, c2, e2, c3, e3, low_memory, predict_final, y_actual

    with _request_lock:
        try:
            data = request.get_json()
            c1 = float(data.get("c1", 0))
            e1 = float(data.get("e1", 0))
            c2 = float(data.get("c2", 0))
            e2 = float(data.get("e2", 0))
            c3 = float(data.get("c3", 0))
            e3 = float(data.get("e3", 0))
            low_memory = bool(int(data.get("low_memory", 0)))

            # Generate target data
            y_actual = [eq(x) for x in x_train]

            # Train the model
            sr.fit(x_train.reshape(-1, 1), y_actual)
            predict_final = sr.predict(x_train.reshape(-1, 1))
            predicted_equation = format_readable_eq(sr._program)

            # Create fresh figure for animation
            fig, axs, lines = create_figure()

            # Animation state
            live_xs_local = []
            live_ys_local = []
            score_ys_local = []

            num_frames = 50  # Reduced for faster loading

            def animate_all(i):
                if i == 0:
                    return lines

                train_index = round(i / num_frames * len(x_train))
                train_index_begin = round((i - 1) / num_frames * len(x_train))

                xs = x_train[:train_index]

                # ACTUAL
                lines[0].set_data(xs, y_actual[: len(xs)])

                if low_memory or len(sr._programs) == 0:
                    # Low memory mode - just show final result
                    ys = sr._program.execute(xs.reshape(-1, 1))
                    lines[1].set_data(xs, ys)
                else:
                    # Map frame to generation
                    idx = round((i * (len(sr._programs) - 1)) / num_frames)
                    idx = min(idx, len(sr._programs) - 1)

                    # PREDICTED - show best at this generation
                    if idx >= len(sr._programs) - 1:
                        ys = predict_final
                    else:
                        fitness_set = [
                            ((prg and abs(prg.fitness_)) or 100)
                            for prg in sr._programs[idx]
                        ]
                        best_prog_idx = fitness_set.index(min(fitness_set))
                        prog = sr._programs[idx][best_prog_idx]
                        ys = prog.execute(x_train.reshape(-1, 1))
                    lines[1].set_data(x_train, ys)

                    # LIVE - progressive
                    xs_slice = x_train[train_index_begin:train_index]
                    if len(xs_slice) > 0:
                        if idx >= len(sr._programs) - 1:
                            ys_slice = predict_final[train_index_begin:train_index]
                        else:
                            ys_slice = ys[train_index_begin:train_index]
                        live_xs_local.extend(xs_slice)
                        live_ys_local.extend(ys_slice)
                    lines[2].set_data(live_xs_local, live_ys_local)

                    # SCORE - R² over time
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
            operator = ""
            match operation[:-1]:
                case "add":
                    operator = " + "
                    valid_op = True
                case "mul":
                    operator = " * "
                    valid_op = True
                case "sub":
                    operator = " - "
                    valid_op = True
                case "div":
                    operator = " / "
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

            return (
                format_readable_eq(first_exp)
                + operator
                + format_readable_eq(second_exp)
            )

    return operation
