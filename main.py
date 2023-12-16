import io
import sys
import random
import math
import base64
import html
import time
import numpy as np
from flask import Flask, Response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
import matplotlib as mpl

#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

app = Flask(__name__)


global_step = 0.0625

c1 = 0.0
e1 = 0.0
c2 = 0.0
e2 = 0.0
c3 = 0.0
e3 = 0.0

domain = 20
domain = round(domain)

low_memory = False

acutal_color = "orange"
predicted_color = "blue"
live_color = "green"


mpl.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
fig = Figure(facecolor='lightskyblue', alpha=0.5)
axis = fig.add_subplot(1,1,1)
axis.set_xlim(-domain/2,domain/2)
axis.set_ylim(-10,10)
line, = axis.plot([], [], lw=2)
line2, = axis.plot([], [], lw=1)

predict_final = []
live_xs = []
live_ys = []
score_ys = []

p_crossover = 0.8             #0.9     0.7     0.8
p_subtree_mutation = 0.03     #0.01     0.1     0.02
p_hoist_mutation = 0.03       #0.01     0.05    0.01
p_point_mutation = 0.02       #0.01     0.1     0.12

p_point_replace = 0.05        #0.05             0.05

parsimony_coefficient = 0.005  #                 0.005

max_samples = 0.6

## 500 pop & 20 gen - cause timeout
pop_size = 800                #1000     5000
gen_amt = 20                  #20       40
tournament_size = 10          #5                20

def _power(x1, x2):
    with np.errstate(over='ignore'):
        return np.where(( (x1 < 100) & (x2 < 20) ), np.power(x1, x2), 0.)
power = make_function(function=_power, name='pow', arity=2)
arctan = make_function(function=np.arctan, name='arctan', arity=1)

def _mape(y, y_pred, w=None):
    y_a = np.array(y)
    y_pred = np.array(y_pred)
    return np.average(np.abs( y_pred - y_a )/y_a, weights=w) * 100
mape = make_fitness(function=_mape, greater_is_better=False, wrap=False)

def _sigmoid(y, y_pred, w=None):
    y_actual = np.array(y)
    y_pred = np.array(y_pred)
    diff = np.abs( y_pred - y_actual )
    sig = (2/(1+np.power(math.e, -diff)))-1
    return (np.average( sig , weights=w))
sigmoid = make_fitness(function=_sigmoid, greater_is_better=False, wrap=False)

def _mae(y, y_pred, w=None):
    return np.average(np.abs(y_pred - y), weights=w)
mae_no_wrap = make_fitness(function=_mae, greater_is_better=False, wrap=False)

#p_crossover, p_subtree_mutation, p_hoist_mutation and p_point_mutation should total to 1.0 or less
sr = SymbolicRegressor(population_size=pop_size, tournament_size=tournament_size,
                           generations=gen_amt, stopping_criteria=0.05,
                           p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                           p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
                           p_point_replace=p_point_replace, metric=mae_no_wrap, #"mse", "rmse" "mean absolute error"
                           const_range=(-5.,5.), random_state=1,
                           parsimony_coefficient=parsimony_coefficient, 
                           function_set=('add', 'mul', 'sin', 'cos', arctan), 
                           low_memory=low_memory, feature_names=["X"],
                           max_samples=max_samples, n_jobs=-1,
                           init_method="half and half")


@app.route("/")
def index_blank():
    print("Blank models requested...")
    global c1,e1,c2,e2,c3,e3,low_memory
    values="0.0:0.0:0.0:0.0:0.0:0.0:0"
    score = 0.0
    #v = values.split(":")
    #c1 = float(v[0])
    #e1 = float(v[1])
    #c2 = float(v[2])
    #e2 = float(v[3])
    #c3 = float(v[4])
    #e3 = float(v[5])
    #low_memory = bool(v[6])

    fig.suptitle('Actual Plot', fontsize=20)
    anim = animation.FuncAnimation(fig, animate_blank, frames=2, interval=40, blit=True, repeat=False)
    actual = anim.to_html5_video().replace('\n', ' ').replace('\r', '')

    fig.suptitle('Predicted Plot', fontsize=20)
    anim = animation.FuncAnimation(fig, animate_blank, frames=2, interval=40, blit=True, repeat=False)
    predicted = anim.to_html5_video().replace('\n', ' ').replace('\r', '')

    live = "Live plot not available in Low Memory Mode."
    if not low_memory:
        fig.suptitle('Live Plot', fontsize=20)
        anim = animation.FuncAnimation(fig, animate_blank, frames=2, interval=40, blit=True, repeat=False)
        live = anim.to_html5_video().replace('\n', ' ').replace('\r', '')

    predicted_equation = "No prediction made yet..."

    return render_template("./index.html", 
        c1=c1,e1=e1, c2=c2,e2=e2, c3=c3,e3=e3, 
        low_mem=low_memory, ps=pop_size, ga=gen_amt, 
        values=values, score=score,
        actual=actual, predicted=predicted, live=live, 
        pr_eq=predicted_equation)


@app.route("/<values>")
def index(values="2.0:2.0:0.5:2.0:6.0:2.0:0"):
    #print("Calculated models requested...")
    global c1,e1,c2,e2,c3,e3,predict_final,low_memory
    v = values.split(":")
    c1 = float(v[0])
    e1 = float(v[1])
    c2 = float(v[2])
    e2 = float(v[3])
    c3 = float(v[4])
    e3 = float(v[5])
    low_memory = bool(int(v[6]))

    fig.suptitle('Actual Plot', fontsize=20)
    anim = animation.FuncAnimation(fig, animate_actual, frames=100, interval=40, blit=True, repeat=False)
    actual = anim.to_html5_video().replace('\n', ' ').replace('\r', '')
    #print("Actual graph is plotted.")

    x_train = np.arange(-domain/2, domain/2, global_step)
    y_train = [eq(x) for x in x_train]
    sr.fit(x_train.reshape(-1,1), y_train)
    #print("Model is trained.")

    #x_temp = np.arange(-domain/2, domain/2, global_step)
    predict_final = sr.predict(x_train.reshape(-1,1))
    #print("Prediction is made.")

    y_actual = [eq(x) for x in x_train]
    score = str( round(sr.score(x_train.reshape(-1,1), y_actual), 3) )
    #score = str(_mape(y_actual, predict_final))[:6]
    pct_score = str( round( 100-(_sigmoid(y_actual, predict_final)*100), 3) )
    #print("Score is calculated.")

    fig.suptitle('Predicted Plot', fontsize=20)
    if low_memory:
        anim = animation.FuncAnimation(fig, animate_low_mem, frames=100, interval=40, blit=True, repeat=False)
    else:
        anim = animation.FuncAnimation(fig, animate_predicted, frames=100, interval=40, blit=True, repeat=False)
    predicted = anim.to_html5_video().replace('\n', ' ').replace('\r', '')
    #print("Prediction graph is plotted.")

    live = "Live plot not available in Low Memory Mode."
    if not low_memory:
        fig.suptitle('Live Plot', fontsize=20)
        anim = animation.FuncAnimation(fig, animate_live_learning, frames=100, interval=40, blit=True, repeat=False)
        live = anim.to_html5_video().replace('\n', ' ').replace('\r', '')
        #print("Live graph is ploted.")

    predicted_equation = sr._program

    #diff = np.abs(np.array(y_actual) - np.array(predict_final))
    #diff/=np.abs(y_actual)
    #pct_score=np.average(diff)*100

    #diff_sig = np.array([(2/(1+np.power(math.e, -y)))-1 for y in diff])
    #score_sig = np.average(diff_sig)*100
    #pct_sig = str(100 - score_sig)[:4] + "%"

    #diff_z = (diff - np.mean(diff)) / np.std(diff)
    #score_z = np.average(diff_z)*100

    #diff_log = np.array([np.log(y) for y in diff])
    #score_log = np.average(diff_log)*100

    #diff_clip = np.array([(y<domain/2 and y>-domain/2 and y) or (y>0 and domain/2) or (y<0 and -domain/2) for y in diff])
    #pct_score_clip = np.average(diff_clip)*100
    #pct_score_clip_n = np.average(diff_clip/domain/2)*100

    #diff_n = diff/diff.max()
    #pct_score_n=np.average(diff_n)*100

    #diff = [round(y,2) for y in diff]
    #diff_n = [round(y,2) for y in diff_n]

    #pct = str(100 - pct_score)[:4] + "%"
    #pct_n = str(100 - pct_score_n)[:4] + "%"
    #pct_c = str(100 - pct_score_clip)[:4] + "%"
    #pct_c_n = str(100 - pct_score_clip_n)[:4] + "%"
    #pct_z = str(100 - score_z)[:4] + "%"
    #pct_log = str(100 - score_log)[:4] + "%"

    return render_template("./index.html", 
        c1=c1,e1=e1, c2=c2,e2=e2, c3=c3,e3=e3, 
        low_mem=low_memory, ps=pop_size, ga=gen_amt, 
        values=values, score=score, pct_score=pct_score,
        actual=actual, predicted=predicted, live=live, 
        pr_eq=predicted_equation, pr_eq_formatted=format_readable_eq(predicted_equation),
        #pct_score=pct_score, pct_score_n=pct_score_n, pct=pct,pct_n=pct_n, 
        #pct_score_clip=pct_score_clip, pct_score_clip_n=pct_score_clip_n, pct_c=pct_c, pct_c_n=pct_c_n,
        #score_z=score_z,pct_z=pct_z, score_log=score_log, pct_log=pct_log,
        #score_sig=score_sig, pct_sig=pct_sig
        )


def format_readable_eq(eq):
    time.sleep(0.1)
    eq = str(eq)
    operation = ""
    for idx, char in enumerate(eq):
        operation += char

        if char == '(':
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
                #print("NON-Valid OP:: " + operation)
                return operation + format_readable_eq(eq[idx+1:-1]) + ")"

            parenthesis = 0
            first_exp = ""
            second_exp = ""
            assign_to_first = True
            for idx_inner, char_inner in enumerate(eq[idx+1:]):
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

            #print("Valid OP("+operator+"):: " + operation)
            return format_readable_eq(first_exp) + operator + format_readable_eq(second_exp)

    #print("EOL:: " + operation)
    return operation

def eq(x):
    #if c1==0 or c2==0 or c3==0:
    #    return 0
    #return math.cos(c1*x) + math.sin(c2*x)
    return math.cos(c1*x)*e1 + math.sin(c2*x)*e2 * math.atan(c3*x)*e3
    #return math.cos(1/c1*x**e1) + math.sin(1/c2*x**e2) * math.tan(1/c3*x**e3)


def animate_blank(i):
    line.set_data([], [])
    return (line,)

def animate_actual(i):
    global line
    line.set_color(acutal_color)
    
    xs = np.arange(0, domain*i/100, global_step)
    xs -= domain/2
    ys = [eq(x) for x in xs]

    line.set_data(xs, ys)
    return (line,)


def animate_low_mem(i):
    global line
    line.set_color(predicted_color)

    if line_ready(i):
        xs = np.arange(0, domain*i/100, global_step)
        xs -= domain/2
        ys = sr._program.execute(xs.reshape(-1,1))

        line.set_data(xs, ys)
    return (line,)

def animate_predicted(i):
    global line
    line.set_color(predicted_color)
    #line.set_color((1-(i/100), i/100, i/100))

    if line_ready(i):
        xs = np.arange(0, domain, global_step)
        xs -= domain/2
        idx = prog_idx(i)

        ys = get_ng_best(idx, xs)

        line.set_data(xs, ys)
    return (line,)

def animate_live_learning(i):
    global live_xs, live_ys, score_ys, line, line2
    line.set_color(live_color)
    line2.set_color("red")

    if line_ready(i):
        xs = np.arange(domain*(i-1)/100, domain*i/100, global_step)
        xs -= domain/2
        live_xs.extend(xs)
        idx = prog_idx(i)

        live_ys.extend( get_ng_best(idx, xs) )
        score_ys.extend( np.full(len(xs), get_capped_score(idx)) )

        line.set_data(live_xs, live_ys)
        line2.set_data(live_xs, score_ys)
    return (line,)


def line_ready(i):
    global line
    if (i==0):
        line.set_data([], [])
        return False
    else:
        return True

def prog_idx(frame):
    return round((frame * (len(sr._programs)-1)) / 100)

def get_ng_best(idx, exec_on):
    if idx == len(sr._programs):
        best_fit = predict_final
        length_of = len(best_fit) - len(exec_on)
        best_fit = best_fit[length_of:]
    else:
        fitness_set = [((prg and abs(prg.raw_fitness_)) or 100) for prg in sr._programs[idx]]
        best_fit_idx = fitness_set.index(min(fitness_set))
        best_fit = sr._programs[idx][best_fit_idx]
        best_fit = best_fit.execute(exec_on.reshape(-1,1))
    return best_fit

def get_capped_score(idx):
    fitness_set = np.array([((prg and abs(prg.fitness_)<9.5 and abs(prg.fitness_)) or 9.5) for prg in sr._programs[idx]])
    best_fit = np.min(fitness_set)
    print(fitness_set);
    return best_fit



#############################
#
# DEPRACTED
#
#############################

def animate_live(i):
    #line.set_color((.1,(i+1)/100,.1))
    line.set_color("green")

    if (i==0):
        live_xs = []
        live_ys = []
        line.set_data([], [])
        return (line,)
    elif (i==99):
        xs = np.arange(0, domain, global_step)
        xs -= domain/2
        ys = predict_final
        #line.set_color("purple")
    else:
        xs = np.arange(0, domain*i/100, global_step)
        xs -= domain/2
        ys = sr.predict(xs.reshape(-1,1))

    line.set_data(xs, ys)
    return (line,)

def animate_live_low_mem(i):
    line.set_color("green")

    if (i==0):
        line.set_data([], [])
        return (line,)
    else:
        xs = np.arange(0, domain*i/100, global_step)
        xs -= domain/2
        ys = sr.predict(xs.reshape(-1,1))
        live_ys.append(ys[-3:])
    print("LIVE::")
    print(live_ys)
    #time.sleep(0.5)

    line.set_data(xs, live_ys)
    return (line,)

def old_format_readable_eq(eq):
    print("RECUR CALL::")
    #time.sleep(1)
    operation = ""
    for idx, char in enumerate(str(eq)):
        if char == '(':
            valid_op = False
            match operation:
                case "add":
                    valid_op = True
                case "mul":
                    valid_op = True
                case "sub":
                    valid_op = True
                case "div":
                    valid_op = True

            if valid_op:
                first_exp = ""
                second_exp = ""
                final_exp = ""
                assign_to_first = True
                in_parenthesis = False
                for idx_inner, char_inner in enumerate(str(eq)[idx+1:]):
                    if char_inner == ' ':
                        continue

                    if assign_to_first:
                        if char_inner == ',':
                            assign_to_first = False
                        else:
                            first_exp += char_inner
                    else:
                        second_exp += char_inner

                    if char_inner == '(':
                        in_parenthesis = True
                    if char_inner == ')':
                        if not in_parenthesis:
                            final_exp = str(eq)[idx_inner+1:]
                            break
                        in_parenthesis = False

                operator = ""
                match operation:
                    case "add":
                        operator = " + " 
                    case "mul":
                        operator = " * " 
                    case "sub":
                        operator = " - " 
                    case "div":
                        operator = " / " 

                out = format_readable_eq(first_exp) + operator + format_readable_eq(second_exp) + format_readable_eq(final_exp)
                print(out)
                return out
            else:
                out = operation + "(" + format_readable_eq(str(eq)[idx+1:-1]) + ")"
                print(out)
                return out

        else:
            operation += char
    out = operation + "(" + format_readable_eq(str(eq)[idx+1:-1]) + ")"
    print(operation)
    return operation


#end