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
import matplotlib.pyplot as plt
import matplotlib as mpl
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

app = Flask(__name__)



########################
### GLOBAL VARIABLES ###
########################

global_step = 0.0625

c1 = 0.0
e1 = 0.0
c2 = 0.0
e2 = 0.0
c3 = 0.0
e3 = 0.0

domain = 20

low_memory = False

predict_final = []
live_xs = []
live_ys = []
score_ys = []



#############################
### GRAPH INTITIALIZATION ###
#############################

acutal_color = "orange"
predicted_color = "blue"
live_color = "green"
score_color = "red"

mpl.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
fig_one = Figure()
fig_one, axs = plt.subplots(4, sharex=True)
fig_one.suptitle('A tale of regression metrics')
fig_one.set_facecolor('lightskyblue')
fig_one.set_figheight(8)
axs[0].set_title("Actual")
axs[1].set_title("Predicted")
axs[2].set_title("Live")
axs[3].set_title("R^2 Score of Determination")
line_act, = axs[0].plot([], [], lw=2, color=acutal_color)
line_pred, = axs[1].plot([], [], lw=2, color=predicted_color)
line_live, = axs[2].plot([], [], lw=2, color=live_color)
line_score, = axs[3].plot([], [], lw=2, color=score_color)
lines = [line_act, line_pred, line_live, line_score]
for idx, ax in enumerate(axs):
    ax.set_xlim(-domain/2,domain/2)
    if idx == len(axs)-1:
        ax.set_ylim(0,100)
    else:
        ax.set_ylim(-10,10)
    ax.grid()



############################
### REGRESSOR PARAMETERS ###
############################

                              #def  |  recc  |  old
p_crossover = 0.8             #0.9      0.7     0.8
p_subtree_mutation = 0.03     #0.01     0.1     0.02
p_hoist_mutation = 0.03       #0.01     0.05    0.01
p_point_mutation = 0.02       #0.01     0.1     0.12

p_point_replace = 0.05        #0.05             0.05

parsimony_coefficient = 0.005  #                0.005
    
max_samples = 0.6   #52.6

## 500 pop & 20 gen - cause timeout
pop_size = 800                #1000     5000
gen_amt = 10                  #20       40
tournament_size = 10          #5                20



#######################
### SCORING METRICS ###
#######################

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
    diff = np.abs( y_pred - y )
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



##################
### APP ROUTES ###
##################

@app.route("/")
def index_blank():
    global c1,e1,c2,e2,c3,e3,low_memory
    values="0.0:0.0:0.0:0.0:0.0:0.0:0"
    score = 0.0
    pct_score = 0.0
    predicted_equation = "No prediction made yet..."
    #full_plot = "No prediction made yet..."

    anim = animation.FuncAnimation(fig_one, animate_blank, frames=1, interval=40, blit=True, repeat=False)
    full_plot = anim.to_html5_video().replace('\n', ' ').replace('\r', '')

    return render_template("./index.html", 
        c1=c1,e1=e1, c2=c2,e2=e2, c3=c3,e3=e3, 
        low_mem=low_memory, ps=pop_size, ga=gen_amt, 
        score=score, pct_score=pct_score,
        pr_eq=predicted_equation, pr_eq_formatted=predicted_equation,
        full_plot=full_plot, samples=max_samples*100)


@app.route("/<values>")
def index(values="2.0:2.0:0.5:2.0:6.0:2.0:0"):
    global c1,e1,c2,e2,c3,e3,predict_final,low_memory
    v = values.split(":")
    c1 = float(v[0])
    e1 = float(v[1])
    c2 = float(v[2])
    e2 = float(v[3])
    c3 = float(v[4])
    e3 = float(v[5])
    low_memory = bool(int(v[6]))

    x_train = np.arange(-domain/2, domain/2, global_step)
    y_actual = [eq(x) for x in x_train]
    sr.fit(x_train.reshape(-1,1), y_actual)

    predict_final = sr.predict(x_train.reshape(-1,1))
    predicted_equation = sr._program

    score = str( round(sr.score(x_train.reshape(-1,1), y_actual), 3) )
    pct_score = str( round( 100-(_sigmoid(y_actual, predict_final)*100), 3) )

    anim = animation.FuncAnimation(fig_one, animate_all, frames=100, interval=40, blit=True, repeat=False)
    full_plot = anim.to_html5_video().replace('\n', ' ').replace('\r', '')


    return render_template("./index.html", 
        c1=c1,e1=e1, c2=c2,e2=e2, c3=c3,e3=e3, 
        low_mem=low_memory, ps=pop_size, ga=gen_amt, 
        score=score, pct_score=pct_score,
        pr_eq=predicted_equation, pr_eq_formatted=format_readable_eq(predicted_equation),
        full_plot=full_plot, samples=max_samples*100)



##########################
### ANIMATOR FUNCTIONS ###
##########################

def animate_blank(i):
    global lines
    return lines

def animate_all(i):
    global lines, live_xs, live_ys, score_ys, low_memory
    xs = np.arange(0, domain*i/100, global_step)
    xs -= domain/2

    ### ACTUAL ###
    ys = [eq(x) for x in xs]
    lines[0].set_data(xs, ys)

    if low_memory:
        ### PREDICTED LOW MEM ###
        ys = sr._program.execute(xs.reshape(-1,1))
        lines[1].set_data(xs, ys)
    else:
        idx = prog_idx(i)

        ### PREDICTED ###
        xs = np.arange(0, domain, global_step)
        xs -= domain/2
        ys = get_ng_best(idx, xs)
        lines[1].set_data(xs, ys)

        ### LIVE ###
        xs = np.arange(domain*(i-1)/100, domain*i/100, global_step)
        xs -= domain/2
        live_xs.extend(xs)
        live_ys.extend( get_ng_best(idx, xs) )
        lines[2].set_data(live_xs, live_ys)

        ### SCORE ###
        score_ys.extend( np.full(len(xs), get_capped_score(idx)) )
        lines[3].set_data(live_xs, score_ys)

    return lines



#########################
### HELPER FUNCTIONS ###
#########################

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
    fitness_set = np.array([((prg and abs(prg.fitness_)<10 and abs(prg.fitness_)) or 10) for prg in sr._programs[idx]])
    best_fit = np.min(fitness_set)*10
    return best_fit


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

            return format_readable_eq(first_exp) + operator + format_readable_eq(second_exp)

    return operation

def eq(x):
    return math.cos(c1*x)*e1 + math.sin(c2*x)*e2 * math.atan(c3*x)*e3



#end