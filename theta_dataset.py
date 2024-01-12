import numpy as np
import heapq
import os

def generate_random_map(width, height, p):
    return np.random.choice([0, 1], (width, height), p=[1-p, p])


def get_random_start_finish_orientation(map_):
    free_positions = np.argwhere(map_ == 0)
    idx = np.random.choice(free_positions.shape[0])
    position = tuple(free_positions[idx])
    orientation = np.random.randint(4)  # Four possible orientations
    return position, orientation


def compute_g_values(start, orientation, map_):
    width, height = map_.shape
    # |orientation space| = 4
    g_values = np.full((4, width, height), np.inf)
    closed_set = np.zeros((4, width, height), dtype=bool)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    open_set = [(0, orientation, start[0], start[1])]
    g_values[orientation, start[0], start[1]] = 0
    
    while open_set:
        g, theta, x, y = heapq.heappop(open_set)
        
        if closed_set[theta, x, y]:
            continue
        closed_set[theta, x, y] = True
        
        # Rotate left and right without moving
        for delta_theta in [-1, 1]:
            new_theta = (theta + delta_theta) % 4
            cost_of_rotation = 1
            new_g = g + cost_of_rotation
            
            if not closed_set[new_theta, x, y] and new_g < g_values[new_theta, x, y]:
                g_values[new_theta, x, y] = new_g
                heapq.heappush(open_set, (new_g, new_theta, x, y))

        dx, dy = directions[theta]
        nx, ny = x + dx, y + dy
        cost_of_movement = 1
        
        if 0 <= nx < width and 0 <= ny < height and not map_[nx, ny] and not closed_set[theta, nx, ny]:
            new_g = g + cost_of_movement
            if new_g < g_values[theta, nx, ny]:
                g_values[theta, nx, ny] = new_g
                heapq.heappush(open_set, (new_g, theta, nx, ny))
                
    return g_values


def compute_matrix(map_):
    s, s_orientation = get_random_start_finish_orientation(map_)
    f, f_orientation = get_random_start_finish_orientation(map_)

    # =============================================
    # PRINT START AND FINISH, START = 2, FINISH = 3
    map_start = np.zeros((4, *map_.shape))
    map_finish = np.zeros((4, *map_.shape))
    map_start[s_orientation, s[0], s[1]] = 1
    map_finish[f_orientation, f[0], f[1]] = 1
    # print(map_start)
    # print(map_finish)
    # =============================================

    while s == f and s_orientation == f_orientation:
        f, f_orientation = get_random_start_finish_orientation(map_)

    g_values_s = compute_g_values(s, s_orientation, map_)
    g_values_f = compute_g_values(f, (f_orientation + 2) % 4, map_)
    optimal_path_value = g_values_s[f_orientation, f[0], f[1]]
    
    all_g_values = g_values_s + np.roll(g_values_f, shift=2, axis=0)

    gt = optimal_path_value / all_g_values
    return np.repeat(map_[np.newaxis, :, :], 4, axis=0), map_start, map_finish, gt


def single_test(count: int = 256):
    maps = []
    starts = []
    finishes = []
    ppms = []

    while count:
        m = generate_random_map(64, 64, 0.3)
        m, s, f, ppm = compute_matrix(m)
        if np.any(np.isnan(ppm)):
            continue
        maps.append(m)
        starts.append(s)
        finishes.append(f)
        ppms.append(ppm)

        count -= 1
    return maps, starts, finishes, ppms


for d, cnt in zip(("train", "val", "test",), (1024, 256, 1024,)):
    top_level_dir = f"TransPath_data_theta/{d}"
    os.makedirs(top_level_dir, exist_ok=True)

    maps, starts, finishes, ppms = single_test(cnt)

    with open(f"{top_level_dir}/maps.npy", "wb") as f:
        np.save(f, np.stack(maps))

    with open(f"{top_level_dir}/starts.npy", "wb") as f:
        np.save(f, np.stack(starts))

    with open(f"{top_level_dir}/goals.npy", "wb") as f:
        np.save(f, np.stack(finishes))

    with open(f"{top_level_dir}/focal.npy", "wb") as f:
        np.save(f, np.stack(ppms))
