from casadi import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def initial(x_start, x_end):
    x_init = np.linspace(x_start[0], x_end[0], N + 1)
    y_init = np.linspace(x_start[1], x_end[1], N + 1)
    theta_init = np.linspace(x_start[2], x_end[2], N + 1)
    v_init = np.full(N, 0.5)
    delta_init = np.full(N, 0.01)
    initial_x = np.empty(N * 5 + 3)
    initial_x[::5] = x_init
    initial_x[1::5] = y_init
    initial_x[2::5] = theta_init
    initial_x[3::5] = v_init
    initial_x[4::5] = delta_init
    return initial_x


def solve(x_initial):
    f = 0
    x = []
    x_lower_bound = []
    x_upper_bound = []
    g = []
    g_lower_bound = []
    g_upper_bound = []
    x_prev = y_prev = theta_prev = []
    v_prev = delta_prev = []
    x_i = y_i = theta_i = []
    v_i = delta_i = []

    for i in range(N + 1):
        if i > 0:
            x_prev = x_i
            y_prev = y_i
            theta_prev = theta_i
            v_prev = v_i
            delta_prev = delta_i

        x_i = SX.sym('x_' + str(i))
        y_i = SX.sym('y_' + str(i))
        theta_i = SX.sym('theta_' + str(i))
        v_i = SX.sym('v_' + str(i))
        delta_i = SX.sym('delta_' + str(i))

        # Set x, y, theta and bounds
        x.extend([x_i, y_i, theta_i])
        if i == 0:
            x_lower_bound.extend(x_0)
            x_upper_bound.extend(x_0)
            f += v_i ** 2 + delta_i ** 2
        elif i == N:
            x_lower_bound.extend(x_goal[0:2])
            x_lower_bound.append(-inf)
            x_upper_bound.extend(x_goal[0:2])
            x_upper_bound.append(inf)
        else:
            # x_lower_bound.extend([-inf, -inf, -np.pi])
            x_lower_bound.extend([-inf, -inf, -inf])
            x_upper_bound.extend([inf, inf, inf])
            f += v_i ** 2 + delta_i ** 2

        # set v, delta and bounds
        if i < N:
            x.extend([v_i, delta_i])
            x_lower_bound.extend([v_min, delta_min])
            x_upper_bound.extend([v_max, delta_max])

        # Dynamics
        if i > 0:
            # Bicycle model
            # F_x_u = sin(delta_prev + theta_prev + T*v_prev*sin(delta_prev))/sin(delta_prev) + x_prev
            # F_y_u = -cos(delta_prev + theta_prev + T*v_prev*sin(delta_prev))/sin(delta_prev) + y_prev
            # F_theta_u = T*v_prev*sin(delta_prev) + theta_prev

            F_x_u = x_prev + (v_prev / delta_prev) * (sin(theta_prev + delta_prev * T) - sin(theta_prev))
            F_y_u = y_prev + (v_prev / delta_prev) * (-cos(theta_prev + delta_prev * T) + cos(theta_prev))
            F_theta_u = theta_prev + delta_prev * T

            g.append(F_x_u - x_i)
            g.append(F_y_u - y_i)
            g.append(F_theta_u - theta_i)
            g_lower_bound.extend([0.0, 0.0, 0.0])
            g_upper_bound.extend([0.0, 0.0, 0.0])

        # Map
        g_map = fmax(x_i, y_i)
        g.append(g_map)
        g_lower_bound.append(map_limit)
        g_upper_bound.append(inf)

        # g.append(y_i)
        # g_lower_bound.append(-inf)
        # g_upper_bound.append(map_upper_limit)

    nlp = {'x': vcat(x), 'f': f, 'g': vcat(g)}
    S = nlpsol('S', 'ipopt', nlp)
    r = S(x0=x_initial, lbg=g_lower_bound, ubg=g_upper_bound, lbx=x_lower_bound, ubx=x_upper_bound)
    return r


N = 20
T = 1.0

x_0 = [6.0, 0.0, np.pi / 2.0]
x_goal = [0.0, 8.0, np.pi]
x_goal = [0.0, 6.0, np.pi / 2.0]

delta_min = -0.15
delta_max = 0.15
v_min = 0.1
v_max = 1.0

c_v = 0.5
c_delta = 1.0 - c_v

map_limit = 5.8
x_map = [map_limit, map_limit, 0]
y_map = [0, map_limit, map_limit]
map_upper_limit = 6.2

x_guess = initial(x_0, x_goal)
r = solve(x_guess)

x_final = r['x'][::5]
y_final = r['x'][1::5]
theta_final = r['x'][2::5]
v_final = r['x'][3::5]
print("delta:", r['x'][4::5])
# print("x:", r['x'][::5])
# print("y:", r['x'][1::5])
# print("theta:", r['x'][2::5])
print("v:", r['x'][3::5])
print("delta:", r['x'][4::5])

map_offset = -10
width = map_limit - map_offset
height = map_limit - map_offset
corner = (map_offset, map_offset)

plot = True
if plot:
    color = [77.0/255, 77.0/255, 77.0/255]
    fig, ax = plt.subplots()
    get_rect = patches.Rectangle(corner, width=width, height=height, color=color)
    ax.add_patch(get_rect)

    ax.plot(x_final, y_final)
    ax.plot(x_final, y_final, 'o')

    # ax.plot(x_map, y_map, 'r')

    plt.xlabel('x')
    plt.ylabel('y')
    ax.axis('equal')

    plt.xlim((-1, 8))
    plt.ylim((-1, 8))
    plt.show()

print("heis")
