import nplab
import matplotlib.pyplot as plt
import laser_align.data_io as d

nplab.datafile.set_current(r"C:\Users\Abhishek\OneDrive - University Of "
                           r"Cambridge\data\drift_mmt"
                           r".hdf5", mode="r")
df = nplab.current_datafile()
# NOTE GAIN 30 FOR THIS! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#ser1 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness'][:, ...]
ser2 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_0'][:, ...]
ser3 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_1'][:, ...]
ser4 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_2'][:, ...]
ser5 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_3'][:, ...]

#ser6 = df['DriftReCentre/DriftReCentre_1/timed_run'][:, ...]
ser7 = df['DriftReCentre/DriftReCentre_1/timed_run_0'][:, ...]
ser8 = df['DriftReCentre/DriftReCentre_1/timed_run_1'][:, ...]
ser9 = df['DriftReCentre/DriftReCentre_1/timed_run_2'][:, ...]
ser10 = df['DriftReCentre/DriftReCentre_1/timed_run_3'][:, ...]

series_dict = {}
for ser in [ser2, ser3, ser4, ser5]:
    series_dict = d.series_maker('Hill Walk {}'.format([ser2, ser3,
                                                        ser4, ser5].index(
        ser)), x=ser[:, 0], y=ser[:, 4], series_dict=series_dict)
for ser in [ser7, ser8, ser9, ser10]:
    series_dict = d.series_maker('Drift {}'.format([ser7, ser8, ser9,
                                               ser10].index(
        ser)), x=ser[:, 0, 0], y=ser[:, 0, 4],
                                 series_dict=series_dict)
d.plot_prettify(series_dict, 'Repeated application of hill walk algorithm',
                'Time/s', 'Intensity at 30 dB gain/AU', x_log=True,
                output='none')
plt.legend().set_visible(False)
plt.show()