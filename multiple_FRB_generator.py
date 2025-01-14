import numpy as np
import matplotlib.pyplot as plt
import fitburst as fb
from copy import deepcopy
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Simulate a burst and save it in fitburst-generic format.")
    parser.add_argument("-i", "--is_dedispersed",
        action="store_true",
        help="Whether the bursts are dedispersed."
    )
    parser.add_argument("-f", "--num_freq",
        type=int,
        default=2 ** 11,
        help="Number of frequency channels."
    )
    parser.add_argument("-t", "--num_time",
        type=int,
        default=28968,
        help="Number of time samples."
    )
    parser.add_argument("-fl", "--freq_lo",
        type=float,
        default=1200.,
        help="Lowest frequency in MHz."
    )
    parser.add_argument("-fh", "--freq_hi",
        type=float,
        default=1800.,
        help="Highest frequency in MHz."
    )
    parser.add_argument("-tl", "--time_lo",
        type=float,
        default=0.,
        help="Earliest time in seconds."
    )
    parser.add_argument("-th", "--time_hi",
        type=float,
        default=300,
        help="Latest time in seconds."
    )
    parser.add_argument("-a", "--amplitude",
        type=float,
        default=0.,
        help="Amplitude of the bursts."
    )
    parser.add_argument("-at1", "--arrival_time1",
        type=float,
        default=75,
        help="Arrival time of the first burst."
    )
    parser.add_argument("-at2", "--arrival_time2",
        type=float,
        default=150,
        help="Arrival time of the second burst."
    )
    parser.add_argument("-at3", "--arrival_time3",
        type=float,
        default=225,
        help="Arrival time of the third burst."
    )
    parser.add_argument("-bw", "--burst_width",
        type=float,
        default=0.001,
        help="Width of the bursts."
    )
    parser.add_argument("-d1", "--dm1",
        type=float,
        default=349.5,
        help="Dispersion measure of the first burst."
    )
    parser.add_argument("-d2", "--dm2",
        type=float,
        default=349.5,
        help="Dispersion measure of the second burst."
    )
    parser.add_argument("-d3", "--dm3",
        type=float,
        default=349.5,
        help="Dispersion measure of the third burst."
    )
    parser.add_argument("-di", "--dm_index",
        type=float,
        default=-2,
        help="Dispersion measure index of the bursts."
    )
    parser.add_argument("-rf", "--ref_freq",
        type=float,
        default=1500,
        help="Reference frequency of the bursts."
    )
    parser.add_argument("-si", "--scattering_index",
        type=float,
        default=-4,
        help="Scattering index of the bursts."
    )
    parser.add_argument("-st", "--scattering_timescale",
        type=float,
        default=0,
        help="Scattering timescale of the bursts."
    )
    parser.add_argument("-sp", "--spectral_index",
        type=float,
        default=2,
        help="Spectral index of the bursts."
    )
    parser.add_argument("-sr", "--spectral_running",
        type=float,
        default=-50,
        help="Spectral running of the bursts."
    )
    parser.add_argument("-s", "--save",
        action="store_true",
        help="Whether to save the data in npz with metadata."
    )
    return parser.parse_args()

def main():
    args = arg_parser()

    freqs = np.linspace(args.freq_lo, args.freq_hi, num = args.num_freq)  
    times = np.linspace(args.time_lo, args.time_hi, num = args.num_time)  
    is_dedispersed = args.is_dedispersed

    # define physical parameters for a dispersed burst to simulate.
    params = {                                                     
        "amplitude"            : [args.amplitude, args.amplitude, args.amplitude],
        "arrival_time"         : [args.arrival_time1,args.arrival_time2, args.arrival_time3 ],
        "burst_width"          : [args.burst_width, args.burst_width, args.burst_width],
        "dm"                   : [args.dm1, args.dm2, args.dm3],
        "dm_index"             : [args.dm_index, args.dm_index, args.dm_index],
        "ref_freq"             : [args.ref_freq, args.ref_freq, args.ref_freq],
        "scattering_index"     : [args.scattering_index, args.scattering_index, args.scattering_index],
        "scattering_timescale" : [args.scattering_timescale, args.scattering_timescale, args.scattering_timescale],
        "spectral_index"       : [args.spectral_index, args.spectral_index, args.spectral_index],
        "spectral_running"     : [args.spectral_running, args.spectral_running, args.spectral_running],
    }  

    num_components = len(params["dm"])

    # define and/or extract parameters.
    new_params = deepcopy(params)

    if is_dedispersed:
        new_params["dm"] = [0.] * num_components

    # define model object for CHIME/FRB data and load in parameter values.
    model_obj = fb.analysis.model.SpectrumModeler(
                freqs,
                times,
                is_dedispersed = is_dedispersed,
                num_components = num_components,
                verbose = True,
            )

    model_obj.update_parameters(new_params)

    # now compute model and add noise.
    model = model_obj.compute_model()
    model += np.random.normal(0., 0.2, size = model.shape)

    # plot.
    plt.pcolormesh(times, freqs, model)
    plt.xlabel("Time (s)")
    plt.ylabel("Observing Frequency (MHz)")
    plt.show()

    #to obtain todays date get the current date and time
    #now = datetime.now()
    # format the current date and time with microseconds
    #formatted_date = now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond:06d}"

    # finally, save data into fitburst-generic format.
    if args.save:
        metadata = {
            "bad_chans" : [],
            "freqs_bin0" : freqs[0],
            "is_dedispersed" : is_dedispersed,
            "num_freq" : args.num_freq,
            "num_time" : args.num_time,
            "times_bin0" : 0.,
            "res_freq" : freqs[1] - freqs[0],
            "res_time" : times[1] - times[0]
        }
        np.savez(
        "archivo_prueba.npz",
        data_full = model,
         metadata = metadata,
        burst_parameters = params,    
        )  


if __name__ == "__main__":
    main()
    
# Generate a synthetic burst

# define dimensions of the data.
# is_dedispersed = True
# num_freq = 2 ** 10
# num_time = 2 ** 11
# freq_lo = 300.
# freq_hi = 500.
# time_lo = 0.
# time_hi = 0.05

# freqs = np.linspace(freq_lo, freq_hi, num = num_freq)  
# times = np.linspace(time_lo, time_hi, num = num_time)  

# # define physical parameters for a dispersed burst to simulate.
# params = {                                                     
#     "amplitude"            : [0],
#     "arrival_time"         : [0.03],
#     "burst_width"          : [0.001],
#     "dm"                   : [349.5],
#     "dm_index"             : [-2],
#     "ref_freq"             : [300],
#     "scattering_index"     : [-4],
#     "scattering_timescale" : [0],
#     "spectral_index"       : [25],
#     "spectral_running"     : [-50],
# }  

# num_components = len(params["dm"])

# # define and/or extract parameters.
# new_params = deepcopy(params)

# if is_dedispersed:
#     new_params["dm"] = [0.] * num_components

# # define model object for CHIME/FRB data and load in parameter values.
# model_obj = fb.analysis.model.SpectrumModeler(
#             freqs,
#             times,
#             is_dedispersed = is_dedispersed,
#             num_components = num_components,
#             verbose = True,
#         )

# model_obj.update_parameters(new_params)

# # now compute model and add noise.
# model = model_obj.compute_model()
# model += np.random.normal(0., 0.2, size = model.shape)

# # plot.
# plt.pcolormesh(times, freqs, model)
# plt.xlabel("Time (s)")
# plt.xlabel("Observing Frequency (MHz)")
# plt.show()

# # finally, save data into fitburst-generic format.
# metadata = {
#     "bad_chans" : [],
#     "freqs_bin0" : freqs[0],
#     "is_dedispersed" : is_dedispersed,
#     "num_freq" : num_freq,
#     "num_time" : num_time,
#     "times_bin0" : 0.,
#     "res_freq" : freqs[1] - freqs[0],
#     "res_time" : times[1] - times[0]
# }

# np.savez(
#     "simulated_data.npz",
#     data_full = model,
#     metadata = metadata,
#     burst_parameters = params,    
# )