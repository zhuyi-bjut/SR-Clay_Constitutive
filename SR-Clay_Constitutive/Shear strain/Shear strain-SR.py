from pysr import PySRRegressor
import numpy as np
import pandas as pd
from astropy import units as u, constants as const
import warnings

warnings.filterwarnings("ignore")
# Import data sets
train = pd.read_excel(r'Shear strain15.xlsx')

features = [x for x in train.columns
            if x not in ['eq', 'eq-1', 's', 'a3', 'K']]
X_train = train[features]
y_train = train['s']

dq = train['dq']
dq = np.array(dq) * u.Pa
q = train['q']
q = np.array(q) * u.Pa
dp = train['dp']
dp = np.array(dp) * u.Pa
p = train['p']
p = np.array(p) * u.Pa
n = train['n']
n = np.array(n) * u.dimensionless_unscaled
G = train['G']
G = np.array(G) * u.Pa
cp = train['cp']
cp = np.array(cp) * u.dimensionless_unscaled
M = train['M']
M = np.array(M) * u.dimensionless_unscaled

s = train['s']
s = np.array(s) * u.dimensionless_unscaled
# Get numerical arrays to fit:
X = pd.DataFrame(dict(
    dq=dq.to("Pa").value,
    q=q.to("Pa").value,
    dp=dp.to("Pa").value,
    p=p.to("Pa").value,
    n=n.value,
    G=G.to("Pa").value,
    cp=cp.value,
    M=M.value,
))
y = s.value

objective = """
function my_custom_objective(tree, dataset::Dataset{T}, options) where {T<:Real}
    # Require root node to be binary, so we can split it,
    # otherwise return a large loss:
    tree.degree != 2 && return T(10000)

    f1 = tree.l
    f1.degree != 2 && return T(10000)
    dp = f1.l
    g1 = f1.r
    dp.degree != 0 && return T(10000)

    f2 = tree.r
    f2.degree != 2 && return T(10000)
    dq = f2.l
    g2 = f2.r
    dq.degree != 0 && return T(10000)

    # Evaluate tree:
    tree_value, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return T(10000)

    # freezing equation in the form of [dq/()]+[dp/()]
    # Assessment of the Plus
    function opadd(t)
        if t.op == 1
            return 0
        else
            return 1111
        end
    end

    # Assessment of division sign
    function opdiv(t)
        if t.op == 4
            return 0
        else
            return 2222
        end
    end

    # Judge the eigenvalue dp
    function feature3(t)
        if !t.constant && t.feature == 3
            return 0
        else
            return 1000
        end
    end

    # Judge the eigenvalue dq
    function feature1(t)
        if !t.constant && t.feature == 1
            return 0
        else
            return 1000
        end
    end

    # constraint expression form
    # Prohibit dq or dp appear in other places.
    function find_dpdq(t)
        if t.degree == 0
            return !t.constant && t.feature in (1, 3)
        elseif t.degree == 1
            return find_dpdq(t.l)
        else
            return find_dpdq(t.l) || find_dpdq(t.r)
        end
    end

    # KGpqcp is prohibited to appear on both sides of +-
    function judge_cat(t)
        h1 = t.l
        h2 = t.r
        if h1.feature in (2, 4, 6, 7) || h2.feature in (2, 4, 6, 7)
            return 1234
        else
            return 0
        end
    end
    # Prevent nesting from appearing
    function judge_bird(t)
        if t.degree == 0
            return !t.constant && t.feature in (2, 4, 6)
        elseif t.degree == 1
            return judge_bird(t.l)
        else
            return judge_bird(t.l) || judge_bird(t.r)
        end
    end  
    # 禁止(cp)^2
    function judge_dog(t)
        if t.degree == 0
            return !t.constant && t.feature == 7
        elseif t.degree == 1
            return judge_dog(t.l)
        else
            return judge_dog(t.l) || judge_dog(t.r)
        end
    end
    # Prevent invalid nesting.1
    function judge_np(t)
    x1 = t.l
    x2 = t.r
        if x1.feature == 4 && x2.feature == 5
            return 999
        elseif x1.feature == 5 && x2.feature == 4
            return 999
        elseif x1.feature == 2 && x2.feature == 5
            return 999
        elseif x1.feature == 5 && x2.feature == 2
            return 999
        else
            return 0
        end
    end
    # Prevent invalid nesting.2
    function judge_qn(t)
    x1 = t.l
    x2 = t.r
        if x1.feature == 2 && x2.feature == 5
            return 999
        elseif x1.feature == 2 && x2.feature == 6
            return 999
        elseif x1.feature == 4 && x2.feature == 6
            return 999
        elseif x1.feature == x2.feature
            return 999        
        else
            return 0
        end
    end
    # Traverse symbol tree, extract all symbols to search_admin
    function search_admin(t, total=0)
        if t.degree == 0
            return total
        elseif t.degree == 1
            total = total + judge_dog(t)*100
        elseif t.degree == 2
            if t.op == 1 || t.op == 2
                total = total + judge_cat(t) + judge_bird(t) * 100
            elseif t.op == 3
                total = total + judge_dog(t.l) * judge_dog(t.r) * 100 + judge_np(t)
            elseif t.op == 4
                total = total + judge_qn(t)
            end 
        end
        # According to the value of degree, it determines which child node to traverse.  
        if t.degree == 1
            total = search_admin(t.l, total)  
        elseif t.degree == 2
            total = search_admin(t.l, total) + search_admin(t.r, total)  
        end
        return total
    end

    # Boundary condition : only for shear strain : tree = 0 when n = 0
    # Evaluate f1:
    f1_value, flag = eval_tree_array(f1, dataset.X, options)
    !flag && return T(10000)
    # Evaluate f2:
    f2_value, flag = eval_tree_array(f2, dataset.X, options)
    !flag && return T(10000)    
    # judge dp component == 0
    if f1_value[1] > 0.0000001
        boundary_loss = 666
    else
        boundary_loss = 0
    end
    # 0 < dp component, 0 < dq component
    dp_value = sum(f1_value)/dataset.n
    dq_value = sum(f2_value)/dataset.n
    if dp_value > 0 && dq_value > 0
        state_loss = 0
    else
        state_loss = 777
    end
    physical_loss = boundary_loss + state_loss    

    prediction = tree_value
    addiv_judge = opadd(tree) + opdiv(f1) + opdiv(f2)
    feature_judge = feature3(dp) + feature1(dq) + T(100)* find_dpdq(g1) + T(100)* find_dpdq(g2)
    catdog_judge = search_admin(g1) + search_admin(g2)
    dimensional_loss = SymbolicRegression.LossFunctionsModule.dimensional_regularization(tree, dataset, options)
    regularization = feature_judge + addiv_judge + catdog_judge + dimensional_loss

    # Impose functional form:
    prediction_loss = sum(abs.(prediction .- dataset.y)) / dataset.n
    return prediction_loss + physical_loss + regularization
end
"""

model = PySRRegressor(
    # procs=4,
    populations=56,
    # ^ Set according to the number of cores
    population_size=500,
    # ^ The larger the population, the greater the diversity.
    ncyclesperiteration=5000,
    # ^ Intergenerational migration.
    niterations=10000,  # Increase me to get better results
    # early_stop_condition=("stop_if(loss, complexity) = loss < 1e-10 && complexity <30"),
    binary_operators=["+", "-", "*", "/"],  # , "special(x, y) = cos(x) * (x + y)"],
    unary_operators=["square"],
    # loss="loss(prediction, target) = abs(prediction - target)",
    loss_function=objective,
    # optimize_probability=5.0,
    maxsize=50,
    maxdepth=50,
    complexity_of_constants=100,
    precision=64,
    # turbo=True,
    progress=True,
    # annealing=True,
    denoise=True,
    # weight_delete_node=3,
    # weight_randomize=1,
    # weight_add_node=10,
    # weight_insert_node=20,
    nested_constraints={
        "square": {"square": 0, "-": 0, "+": 0},
        # "cube": {"square": 0, "cube": 0},
        # "sqrt": {"^": 0,"square":0,"sqrt": 0},
        "+": {"+": 1, "-": 1},
        # "*": {"*": 1},
        "-": {"-": 1, "+": 1},
        # "exp": { "exp": 0,"log":0,"square":0,"sqrt": 0},
        # "log": { "exp": 0,"log":0,"square":0,"sqrt": 0},
        # "^": { "^": 0,"square":0,"sqrt": 0},
        "/": {"/": 1},
    },
    # Amount to penalize dimensional violations:
    dimensional_constraint_penalty=10 ** 5,
    # select_k_features=4,
)
model.fit(
    X,
    y,
    X_units=["Pa", "Pa", "Pa", "Pa", "1", "Pa", "1", "1"],
    y_units=["1"]
)
print(model)