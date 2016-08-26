import microscope as micro
import time
import numpy as np
import yaml

positions = [30000, -30000, 10000, -10000, 5000, -5000, 1000, -1000, 500,
             -500, 100, -100]

stage = micro.Stage('../configs/config.yaml', override=True, backlash=[0, 0,
                                                                       0])
results = {}

print "Starting in 10 seconds."
time.sleep(10)
print "started"

for axis in [0, 1, 2]:
    mean_results = np.zeros((1, len(positions), 2))
    for i in range(3):
        sub_results = []
        for position in positions:
            move_by = [0, 0, 0]
            move_by[axis] = position
            stage.move_rel(move_by)
            start = time.time()
            raw_input('Press enter when motion done.')
            finish = time.time()
            diff = finish - start
            sub_results.append([position, diff])
        sub_results = np.array([sub_results])
        mean_results += sub_results
    mean_results /= 3.
    results[axis] = mean_results

print results
with open('motor_speed.yaml', 'w') as f:
    yaml.dump(results, f, default_flow_style=True)



