from numpy import *
# y = mx + b
# m is slope, b is y-intercept
# this is cost function or error function
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

# magic happen here
# learning_rate : it indicate that how big step it takes to the downhill with gradient decent.
def step_gradient(b_current, m_current, points, learningRate):
    # start point for gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    # calculate the partial derivative
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # compute the partial derivatives of error functoins
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    # update b and m using our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    # gradient decent
    for i in range(num_iterations):
        # update b and m with more accurate b and m by performing gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    # step 1: collect our data;
    points = genfromtxt("data.csv", delimiter=",")

    # step 2: define hyper parameters
    ## how fast should our model converge?
    learning_rate = 0.0001
    ## slop formula (y = mx + b)
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."

    # step 3: train our model
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
