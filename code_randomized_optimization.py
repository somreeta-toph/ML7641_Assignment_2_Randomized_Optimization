import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import time

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness



def Random_Hill_Climb(problem,max_att=10,max_iters=1000,datasetName="8-Queens"):
    best_state, best_fitness, curve = mlrose.random_hill_climb(problem,
					      max_attempts = max_att, max_iters = max_iters, random_state = 1, curve=True)
        
    print('The best state, random hill climb : ', best_state)
    print('The fitness at best state, random hill climb: ', best_fitness)
    

    xdata=[]
    ydata=[]
    for i in range(0,max_iters,20):
        best_state, best_fitness, curve = mlrose.random_hill_climb(problem,
					      max_attempts = max_att, max_iters = i, random_state = 1, curve=True)
        xdata.append(i)
        ydata.append(best_fitness)

    Plot_Fitness_Only_One(xdata,ydata,algo_name="RHC",dataset=datasetName)
    return (xdata,ydata)
    

def Sim_Ann(problem, schedule, init_state,max_att=10,max_iters=1000,datasetName="8-Queens"):
    # Solve problem using simulated annealing
    best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = max_att, max_iters = max_iters,
                                                          init_state = init_state, random_state = 1, curve=True)

    print('The best state, simulated annealing : ', best_state)
    print('The fitness at best state, simulated annealing: ', best_fitness)
    #print("Sim curve", curve)

    
    xdata=[]
    ydata=[]
    for i in range(0,max_iters,20):
        best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = max_att, max_iters = i,
                                                          init_state = init_state, random_state = 1, curve=True)
        
        xdata.append(i)
        ydata.append(best_fitness)
    
    
    Plot_Fitness_Only_One(xdata,ydata,algo_name="Simulated_Annealing",dataset=datasetName)
    return (xdata,ydata)




def Genetic_Alg(problem, max_att=10,max_iters=1000,datasetName="8-Queens",mute=0.2):
    best_state, best_fitness, curve = mlrose.genetic_alg(problem, mutation_prob = mute, 
					      max_attempts = max_att, max_iters = max_iters, random_state = 1, curve=True)
        
    print('The best state, genetic alg : ', best_state)
    print('The fitness at best state, genetic alg: ', best_fitness)

    xdata=[]
    ydata=[]
    for i in range(0,max_iters,20):
        best_state, best_fitness, curve = mlrose.genetic_alg(problem, mutation_prob = mute, 
					      max_attempts = max_att, max_iters = i, random_state = 1, curve=True)
        
        xdata.append(i)
        ydata.append(best_fitness)
    
    
    Plot_Fitness_Only_One(xdata,ydata,algo_name="Genetic_Algorithm",dataset=datasetName)
    return (xdata,ydata)


def Mimic(problem, max_att=10,max_iters=1000,keep=0.2,datasetName="8-Queens"):
    best_state, best_fitness, curve = mlrose.mimic(problem,keep_pct=keep,
                                                   max_attempts = max_att, max_iters = max_iters, random_state = 1, curve=True)
        
    print('The best state, MIMIC : ', best_state)
    print('The fitness at best state, MIMIC: ', best_fitness)

    xdata=[]
    ydata=[]
    for i in range(0,max_iters,20):
        best_state, best_fitness, curve = mlrose.mimic(problem,keep_pct=keep,
                                                   max_attempts = max_att, max_iters = i, random_state = 1, curve=True)
        
        xdata.append(i)
        ydata.append(best_fitness)
    
    
    Plot_Fitness_Only_One(xdata,ydata,algo_name="MIMIC",dataset=datasetName)
    return (xdata,ydata)
    

def Plot_Fitness_Only_One(xdata,ydata,algo_name="",dataset=""):
    figure_name = algo_name + "Dataset-" + str(dataset) + ".jpg"
    
    #fig, ax = plt.subplots()
    title = " : Fitness Curve - "+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Iterations --> ")
    plt.ylabel("Fitness --> ")
    plt.plot(xdata,ydata)
    plt.legend([algo_name])
    plt.savefig(figure_name)


def Plot_Fitness(xdata1, ydata1, ydata2, ydata3, ydata4, algo_name="", dataset="",legends=[]):
    figure_name = algo_name + "Dataset-" + str(dataset) + ".jpg"
    
    #fig, ax = plt.subplots()
    title = " : Fitness Curve - "+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Iterations --> ")
    plt.ylabel("Fitness --> ")
    plt.plot(xdata1,ydata1,color='#1f77b4') #deault blue
    plt.plot(xdata1,ydata2,color='darkorange')
    plt.plot(xdata1,ydata3,color='green')
    plt.plot(xdata1,ydata4,color='red')
    if len(legends)==0:
        plt.legend([algo_name])
    else:
        plt.legend(legends)
        
    plt.savefig(figure_name)
    
    


def Optimize(dataset=1):
    #Get problem

    if dataset==1:
        print("8Q:")
        problem, schedule, init_state = GetDataSetQueens()
        x1,y1 = Random_Hill_Climb(problem)
        x2,y2 = Sim_Ann(problem, schedule, init_state)        
        x3,y3 = Genetic_Alg(problem)
        x4,y4 = Mimic(problem)
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset="8-Queens",legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])

    elif dataset==2:
        print("TSOP:")
        problem, schedule, init_state = GetTravelingSalesmanDataSet()
        x1,y1 = Random_Hill_Climb(problem,datasetName="TSOP")
        x2,y2 = Sim_Ann(problem, schedule, init_state,datasetName="TSOP")        
        x3,y3 = Genetic_Alg(problem,datasetName="TSOP")
        x4,y4 = Mimic(problem,datasetName="TSOP")
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset="Traveling_Salesman",legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])

    elif dataset==3:
        print("1 MAX")
        problem, schedule, init_state = GetOneMaxDataSet()
        datasetName="OneMax"
        x1,y1 = Random_Hill_Climb(problem,max_iters=200,datasetName=datasetName)
        x2,y2 = Sim_Ann(problem, schedule, init_state,max_iters=200,datasetName=datasetName)        
        x3,y3 = Genetic_Alg(problem,max_iters=200,datasetName=datasetName,mute=0.23)
        x4,y4 = Mimic(problem,max_iters=200,datasetName=datasetName)
        print("plotting")
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset=datasetName,legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])

    elif dataset==4:
        print("Knapsack:")
        problem, schedule, init_state = GetKnapsackDataSet()
        datasetName="Knapsack"
        x1,y1 = Random_Hill_Climb(problem,datasetName=datasetName)
        x2,y2 = Sim_Ann(problem, schedule, init_state,datasetName=datasetName)        
        x3,y3 = Genetic_Alg(problem,datasetName=datasetName)
        x4,y4 = Mimic(problem,datasetName=datasetName)
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset=datasetName,legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])

    elif dataset==5:
        print("4 peaks")
        problem, schedule, init_state = GetKnapsackDataSet()
        datasetName="FourPeaks"
        x1,y1 = Random_Hill_Climb(problem,max_att=100,datasetName=datasetName)
        x2,y2 = Sim_Ann(problem, schedule, init_state,max_att=100,datasetName=datasetName)        
        x3,y3 = Genetic_Alg(problem,max_att=100,datasetName=datasetName)
        x4,y4 = Mimic(problem,max_att=100,datasetName=datasetName)
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset=datasetName,legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])

    elif dataset==6:
        print("KColor")
        problem, schedule, init_state = GetKColor()
        datasetName="KColor"
        x1,y1 = Random_Hill_Climb(problem,datasetName=datasetName)
        x2,y2 = Sim_Ann(problem, schedule, init_state,datasetName=datasetName)        
        x3,y3 = Genetic_Alg(problem,datasetName=datasetName)
        x4,y4 = Mimic(problem,keep=0.3,datasetName=datasetName)
        Plot_Fitness(x1, y1, y2, y3, y4, algo_name="", dataset=datasetName,legends=["RHC","Simulated_Annealing","Genetic_Algo","MIMIC"])
        
        
        
    
        
    
def GetDataSetQueens():
    fitness_cust = mlrose.CustomFitness(queens_max)
    fitness = mlrose.Queens()
    problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                                     maximize = False, max_val = 8)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    return problem, schedule, init_state

def GetTravelingSalesmanDataSet():
    # Create list of city coordinates
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords = coords_list)

    # Define optimization object
    problem = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords,
                            maximize=False)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    return problem, schedule, init_state

def GetOneMaxDataSet():
    fitness = mlrose.OneMax()
    problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                                     maximize = True, max_val = 8)
        # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    return problem, schedule, init_state


def GetKnapsackDataSet():
    weights = [10, 5, 2, 8, 15, 1, 9, 20]
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    max_weight_pct = 0.5
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness,
                                     maximize = True, max_val = 20)
    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    return problem, schedule, init_state

def FourPeaks():
    fitness = mlrose.FourPeaks(t_pct=0.15)
    problem = mlrose.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.random.randint(2,size=100)

    # Define the problem
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)

    # Define the optimization problem
    optimization_problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)


    return problem, schedule, init_state

def GetKColor():
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    fitness = mlrose.MaxKColor(edges)
    problem = mlrose.DiscreteOpt(length = 7, fitness_fn = fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2, size=7)
    schedule = mlrose.ExpDecay()

    return problem, schedule, init_state

def four_peaks_iterations():

    fitness = mlrose.FourPeaks(t_pct=0.15)
    problem = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness, maximize=True, max_val=2)
    init_state = np.random.randint(2,size = 50)
    schedule = mlrose.GeomDecay()
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters=10000, init_state = init_state, curve=True)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters=10000, init_state = init_state, curve=True)
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts = 100, max_iters=10000, curve=True)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem,pop_size=500,max_attempts = 100, max_iters=10000, curve=True)
	
    plt.figure()
    plt.plot(fitness_curve_sa,label='SA')
    plt.plot(fitness_curve_rhc,label='RHC')
    plt.plot(fitness_curve_ga,label='GA')
    plt.plot(fitness_curve_mimic,label='MIMIC')
    plt.legend()
    plt.ylabel('Fitness Value')
    plt.xlabel('Number of Iterations')
    plt.title('Fitness Value vs. Number of Iterations (4 Peaks)')
    plt.savefig('4_peaks_iterations.png')
    return
    
# Define the fitness function
def fitness_fn(state):
    zeros = np.count_nonzero(state == 0)
    ones = np.count_nonzero(state == 1)
    if zeros == len(state) or ones == len(state):
        return len(state)
    else:
        return max(zeros, ones)


def OptimizeNeuralNet(algo=1):
    X,y,name = GetDataNeuralNet()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, random_state = 3)

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    # Initialize neural network object and fit object
    if algo == 1:
        
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'random_hill_climb', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
				 random_state = 3)

    elif algo == 2:
        
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'simulated_annealing', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
				 random_state = 3)


    elif algo == 3:
        
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'genetic_alg', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
				 random_state = 3)


    elif algo == 4:
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'mimic', max_iters = 1000,
                                 bias = True, is_classifier = True, learning_rate = 0.0001,
                                 early_stopping = True, clip_max = 5, max_attempts = 100,
				 random_state = 3)

    #start = time.time()
    nn_model1.fit(X_train_scaled, y_train_hot)
    #end = time.time()
    #print("train clock time in us: ", (end-start)*1000000)

    # Predict labels for train set and assess accuracy
    start = time.time()
    y_train_pred = nn_model1.predict(X_train_scaled)
    end = time.time()
    print("train clock time in us: ", (end-start)*1000000)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print('Training accuracy: ', y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print('Test accuracy: ', y_test_accuracy)


def GetDataNeuralNet(dataset = 1):
    """
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    """
    if dataset == 1:
        name = "fetal_health"
    else:
        name = "mobile_price"
    name = name + ".csv"
    df = pd.read_csv(name)
    data = df.to_numpy() # rows and columns just like .csv
    X = data[:,0:-1]
    y = np.transpose(data)[-1]
    #print("data",data)
    #print("X",X)
    #print("y",y)
    return (X,y,name)


    


if __name__=="__main__":
    print("One day you'll look back on this and smile. \
    There will be tears, but they will be tears of joy")
    #Optimize(1)
    #Optimize(2)
    #Optimize(3)
    #Optimize(4)
    #Optimize(5)
    #four_peaks_iterations()
    Optimize(6)
    #OptimizeNeuralNet(1)
    #OptimizeNeuralNet(2)
    #OptimizeNeuralNet(3)
    #OptimizeNeuralNet(4)
