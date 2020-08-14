
import numpy as np
import matplotlib.pyplot as plt
import terminalplot as tplt
import argparse

FORM = "sine"
PERIOD = 1
SHIFT = 0

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--form', type=str, default=FORM)
    parser.add_argument('--min', type=float, default=0)
    parser.add_argument('--max', type=float, default=1)
    parser.add_argument('--period', type=float)
    parser.add_argument('--frequency', type=float, default=None)
    parser.add_argument('--graph', type=str, default="graphic")
    parser.add_argument('--phaseshift', type=float, default=SHIFT)
    
    return parser.parse_args()

def frequency_to_period(frequency):
    return 1/frequency

def wave(form, min, max, period, step, samplerate, samplesize, phaseshift):
    axis = 0
    input = ((np.pi*2/period) / samplerate) * (step - samplerate*phaseshift) #x value mapped into 2*pi
    range = max - min
    if form == "sine":
        return sine(range, axis, input)
    elif form == "square":
        if sine(range, axis, input) >= axis:
            return max
        elif sine(range, axis, input) < axis:
            return min
    elif form == "triangle":
        return triangle(range, axis, input)
    else: 
        raise Exception("wrong value on --form")

def sine(range, axis, input):
    if range <= 0:
        raise Exception("wrong range of min-max")
    return (np.sin(input)*0.5*range)

def triangle(range, axis, input):
    temp_period = (np.pi*2)*np.floor(input/(np.pi*2))
    if input%(np.pi*2) < np.pi*0.5:
        return((range*0.5)/(0.5*np.pi)*(input-temp_period))+axis
    elif input%(np.pi*2) < np.pi*1.5:
        return((-range)/(np.pi)*(input-(0.5*np.pi)-temp_period))+range*0.5
    elif input%(np.pi*2) < np.pi*2:
        return((range*0.5)/(0.5*np.pi)*(input-(1.5*np.pi)-temp_period))-range*0.5

def generate_value(step, samplerate, form, _min, _max, period, samplesize, phaseshift):
    #array for graph plot
    x_array = []
    temp_y_array = []
    

     # outputs y_value as each x_value (step)
    for x in range(samplesize):
        x_array.append(step)
        temp_y_array.append(wave(form, _min, _max, period, step, samplerate, samplesize, phaseshift))         
        step += 1

     # coordinate the min/max value of the graph
    practical_range = max(temp_y_array)-min(temp_y_array)
    _range = _max - _min
     #find the proper axis    
    if min(temp_y_array) >= 0:
        axis = _min

   
    y_array = [i * (_range/practical_range) + _min -min(temp_y_array) for i in temp_y_array]
    arrays = [x_array, y_array]
 

    return arrays

def generate_graph(arrays, graph):   
    #get x, y values to draw a graph
    x_array = arrays[0]
    y_array = arrays[1]

    #determine if show graph or not
    if graph == "graphic": #generates new graphic window with matplotlib
        plt.scatter(x_array, y_array, color="green", marker="1", s=30)
        plt.xlabel('x_axis')
        plt.ylabel('y_axis')
        plt.title('plot')
        plt.show()
    elif graph == "terminal": #draw graph in terminal
        tplt.plot(x_array, y_array)
    elif graph == None:
        pass

def main():
    args = get_arguments()

    #set some arbitary values for step and samplerate
    # these values will be replaced with real values in generate.py
    step = 0
    samplerate = 160
    samplesize = 320
    if args.frequency is not None:
        if args.period is not None:
            raise ValueError("Frequency and Period both assigned. Assign only one of them.")
        else: #change frequency into period
            PERIOD = frequency_to_period(args.frequency)
    else: #get period input value
        try:
            PERIOD = args.period
            if PERIOD is None:
                raise TypeError
        except TypeError:
            PERIOD = 1

    generate_graph(generate_value(step, samplerate, args.form, args.min, args.max, PERIOD, samplesize, args.phaseshift), args.graph)

if __name__ == '__main__':
    main()