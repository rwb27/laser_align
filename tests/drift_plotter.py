import nplab
import laser_align.data_io as d

nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\drift_mmt.hdf5", mode="r")
df = nplab.current_datafile()

ser1 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness'][:, ...]
ser2 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_0'][:, ...]
ser3 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_1'][:, ...]
ser4 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_2'][:, ...]
ser5 = df['DriftReCentre/DriftReCentre_1/hill_walk_brightness_3'][:, ...]

ser6 = df['DriftReCentre/DriftReCentre_1/timed_run'][:, ...]
ser7 = df['DriftReCentre/DriftReCentre_1/timed_run_0'][:, ...]
ser8 = df['DriftReCentre/DriftReCentre_1/timed_run_1'][:, ...]
ser9 = df['DriftReCentre/DriftReCentre_1/timed_run_2'][:, ...]
ser10 = df['DriftReCentre/DriftReCentre_1/timed_run_3'][:, ...]

series_dict = {}
for ser in [ser1, ser2, ser3, ser4, ser5]:
    series_dict = d.series_maker('hill_walk_{}'.format([ser1, ser2, ser3,
                                                        ser4, ser5].index(
        ser)), x=ser[:, 0], y=ser[:, 4], series_dict=series_dict)
for ser in [ser6, ser7, ser8, ser9, ser10]:
    series_dict = d.series_maker('timed_{}'.format([ser6, ser7, ser8, ser9, ser10].index(
        ser)), x=ser[:, 0, 0], y=ser[:, 0, 4],
                                 series_dict=series_dict)
d.plot_prettify(series_dict, 'Brightness vs time, gain=30', 'time/s',
                'brightness/AU', x_log=True)