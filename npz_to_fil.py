import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
import glob
from astropy import units as apu
from astropy.time import Time
from your.formats.filwriter import make_sigproc_object
from your import Your

def write_header_for_fil(filename, telescope, nchans, foff, fch1, tsamp, tstart, nbits):
    """
    Write the data to the filterbank file

    Parameters
    ----------
    sigpyproc_object : object
    filename : str
        Name of the file
    data : numpy array
        Data to write to the file

    Returns
    -------
    sigpyproc_object : object
    """
    sigproc_object = make_sigproc_object(
        rawdatafile=filename,
        source_name=telescope,
        nchans=nchans, 
        foff=foff,
        fch1=fch1,
        tsamp=tsamp,
        tstart=tstart,
        src_raj=230909.6,  # HHMMSS.SS
        src_dej=484201.0,  # DDMMSS.SS
        machine_id=0,
        nbeams=0,
        ibeam=0,
        nbits=nbits,
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=1,
        data_type=0,
        az_start=-1,
        za_start=-1,
    )
    sigproc_object.write_header(filename)
    return sigproc_object


def write_data_to_fil(sigpyproc_object, filename, data):
    """
    Write the data to the filterbank file

    Parameters
    ----------
    sigpyproc_object : object
    filename : str
        Name of the file
    data : numpy array
        Data to write to the file

    Returns
    -------
    sigpyproc_object : object
    """
    sigpyproc_object.append_spectra(spectra = data, filename = filename)
    return 


def get_output_path(path_output, filename):
    """
    Get the output path

    Parameters
    ----------
    path_output : str
        Path to the output
    filename : str
        Name of the file

    Returns
    -------
    path_save_fil : str
    """
    #fullpath_without_space = Path(full_path.as_posix().replace(" ", "_"))
    full_path = path_output / filename
    full_path = os.path.splitext(full_path)[0]
    path_save_fil = path_output.joinpath(full_path + ".fil")
    #path_save_fil_str = "/home/emi/Descargas.fil"
    #path_save_fil = Path(path_save_fil_str)

    return path_save_fil

def get_frequencies(bandpass, nchannels):
    
    return np.linspace(*bandpass, nchannels, endpoint=False)[::-1]

def parser_args():

    parser = argparse.ArgumentParser(description='Convert the data from logfile to a .fil file')
    parser.add_argument('--path_data', type=str, default=os.getcwd(), help='path to the data')
    parser.add_argument('--path_output', type=str, default=os.getcwd(), help='path to the output')
    parser.add_argument('--filename', type=str, nargs='+', default=glob.glob('*'),
                        help='Specify filenames or use the default (all files in the current directory)')
    parser.add_argument('--telescope', type=str, default = 'ARTE', help='name of the telescope')
    parser.add_argument('--bandpass', type=float, nargs=2, default = [1200, 1800], help='frequencies in MHz')
    parser.add_argument('--freq_chann0', type=int, default = 0, help='reference frequency in MHz')
    parser.add_argument('--time_resolution', type=float, default = 0.01, help='time resolution in seconds')
    parser.add_argument('--nbits', type=int, default = 32, help='number of bits')
    parser.add_argument('--nchannels', type=int, default = 2048, help='number of channels')
    parser.add_argument('--flip', action='store_true', help='flip the data in the frequency axis')
    parser.add_argument('--padding', type=int, default=0, help='number of rows to add to the data')
    parser.add_argument('--remove_samples', type=int, default=0, help='number of samples to remove from the data')

    return parser.parse_args()

def main():

    args = parser_args()
    path_data = Path(args.path_data)
    path_save = Path(args.path_output)
    filenames = args.filename
    filenames.sort()
    telescope = args.telescope
    bandpass = args.bandpass 
    freq_chann_0 = args.freq_chann0 
    time_resolution = args.time_resolution 
    nbits = args.nbits
    nchannels = args.nchannels
    bandwidth = bandpass[0] - bandpass[1]
    for filename in filenames:
        #full_path, start_time = get_date_from_logfile(path_data, filename)
        frequencies = get_frequencies(bandpass, nchannels)
        path_save_fil = get_output_path(path_data, filename)
        start_time = 0
        
        sigproc_object = write_header_for_fil(
            path_save_fil.as_posix(),
            telescope,
            nchannels,
            bandwidth/nchannels,
            frequencies[freq_chann_0],
            time_resolution,
            start_time,
            nbits
        )     

        data = np.load(filename)['data_full']
        #print(np.shape(data))
        #plt.imshow(data, origin='lower')
        #plt.show()
        data=np.transpose(data)
        print(np.shape(data))
        data = data.astype('float' + str(nbits))
        
        write_data_to_fil(sigproc_object, path_save_fil.as_posix(), data)
        print('Final shape:', data.shape) 
        print("File:", filename, "completed!")

if __name__ == '__main__':
    main()
