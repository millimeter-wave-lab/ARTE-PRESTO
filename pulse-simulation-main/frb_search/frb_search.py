# script to search for FRBs in data using the PRESTO pipeline

import os
import glob
import subprocess
import argparse
from astropy import units as u
import frb_search.plot_waterfall as pw
import frb_search.candidates as cands
from pathlib import Path
from .telegram_bot import send_message
from .config import config
import frb_search.converter as converter
import PyPDF2
#from memory_profiler import profile
# define functions


def merge_pdfs(pdfs, output_file):
    pdf_writer = PyPDF2.PdfMerger()
    for pdf in pdfs:
        pdf_writer.append(pdf)
    with open(output_file, "wb") as out:
        pdf_writer.write(out)


def find_pdf_in_subdir(subdir):
    full_subdir_path = subdir  #ARTE version
    if not os.path.exists(full_subdir_path):
        print('Path does not exist.')
        return None
    for file in os.listdir(full_subdir_path):
        if file.endswith('.pdf'):
            return os.path.join(full_subdir_path, file)
    return None


def collect_pdfs(root_dir):
    pdf_files = []
    for subdir in os.listdir(root_dir):
        full_subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_subdir_path):
            pdf_file = find_pdf_in_subdir(full_subdir_path)
            if pdf_file:
                pdf_files.append(pdf_file)
    return pdf_files


def delete_pdfs(pdfs):
    for pdf_file in pdfs:
        os.remove(pdf_file)

def write_summary(summary_file, file_name, df, df_, num_plotted, downsample, nsub, m, sigma,
                   dt, plot_window, timeseries, spectral_average, start_dm, num_dms, dm_step,
                     num_to_plot, stokes):
    with open(summary_file, 'w') as f:
            f.write('Summary of the FRB search pipeline\n')
            f.write('Filename: ' + file_name + '\n')
            f.write('Number of candidates after filtering: ' + str(len(df_)) + '\n')
            f.write('Number of candidates plotted: ' + str(num_plotted) + '\n')
            f.write('Total candidates obtained by PRESTO: ' + str(len(df)) + '\n')
            f.write('Downsampling factor: ' + str(downsample) + '\n')
            f.write('Number of subbands: ' + str(nsub) + '\n')
            f.write('Maximum width of pulse: ' + str(m) + '\n')
            f.write('Sigma threshold: ' + str(sigma) + '\n')
            f.write('Time in seconds to separate events: ' + str(dt) + '\n')
            f.write('Time in seconds to plot around each candidate: ' + str(plot_window) + '\n')
            f.write('Time series plotted: ' + str(timeseries) + '\n')
            f.write('Spectral average plotted: ' + str(spectral_average) + '\n')
            f.write('Start DM: ' + str(start_dm) + '\n')
            f.write('Number of DMs: ' + str(num_dms) + '\n')
            f.write('DM step size: ' + str(dm_step) + '\n')
            f.write('Number of candidates to plot: ' + str(num_to_plot) + '\n')
            f.write('Stokes parameter: ' + str(stokes) + '\n')
    return

def parser_args():
    '''
    Function to parse the command line arguments
    
    Input:
        None
    Output:
        args: object containing the command line arguments
    '''
    parser = argparse.ArgumentParser(description='Search for FRBs in data using the PRESTO pipeline')
    parser.add_argument('--filename', nargs='+' ,type=str, default=glob.glob('*'), help='name of the .fil file')
    parser.add_argument('--dm_start', type=float, help='starting DM to search', default = 0)
    parser.add_argument('--num_dms', type=int, help='number of DMs to search', default = 50)
    parser.add_argument('--dm_step', type=float, help='DM step size', default = 10)
    parser.add_argument('--downsample', type=int, default=1, help='downsampling factor')
    parser.add_argument('--nsub', type=int, help='number of subbands', default = 2048)
    parser.add_argument('-m', type=float, help='maximum width of pulse to search for in seconds', default = 0.05)
    parser.add_argument('--sigma', type=float, help='sigma threshold to search for candidates', default = 8)
    parser.add_argument('--dt', type=float, help='time in seconds to separate events', default = 0.5)
    parser.add_argument('--plot_window', type=float, help='time in seconds to plot around each candidate', default = 1)
    parser.add_argument('--timeseries', action='store_false', help='whether to plot the time series')
    parser.add_argument('--spectral_average', action='store_true', help='whether to plot the spectral average')
    parser.add_argument('--stokes', type=int, default=0, help='Stokes parameter to plot (0, 1, 2, 3) for (I, Q, U, V)')
    parser.add_argument('--num_to_plot', type=int, default=150, help='number of candidates to plot')
    parser.add_argument('--mask', action='store_true', help='if True then skip the rfifind step')
    parser.add_argument('--save', action='store_false', help='whether to save the plots')
    parser.add_argument('--convert_to_fil', action='store_false', help='convert ARTE logfiles to .fil files')
    parser.add_argument('--bandpass', type=float, nargs=2, default=[1200, 1800], help='bandpass to apply to the data (low, high) in MHz')
    parser.add_argument('--time_resolution', type=float, default=0.01, help='time resolution of the data in seconds')
    parser.add_argument('--nchannels', type=int, default=2048, help='number of frequency channels in the data')
    parser.add_argument('--flip_data', action='store_false', help='flip the data for conversion in frequency')
    parser.add_argument('--padding', type=int, default=0, help='padding to add to the data in frequency channels')
    parser.add_argument('--remove_samples', type=int, default=0, help='number of samples to remove from the start of the data for conversion')
    parser.add_argument('--remove_fil', action='store_false', help='remove the .fil file after conversion and processing')
    parser.add_argument('--prepdata', action='store_false', help='whether to run the PRESTO prepdata command instead of prepsubband')
    parser.add_argument('--wrap_pdfs', action='store_true', help='whether to wrap the pdfs into a single pdf file')
    parser.add_argument('--output_dir', type=str, default='/media/transferencia/4TBdisk/PRESTO/analized_data', help='directory where to process the fils and store the output files')
    parser.add_argument('--logfiles_dir', type=str, default=os.getcwd(), help='directory where the logfiles are stored')
    return parser.parse_args()
    

#@profile
def main():
    args = parser_args()
    # run the PRESTO pipeline
    # define the name of the .fil file

    filenames = args.filename
    
    filenames.sort()
    
    start_dm = args.dm_start
    num_dms = args.num_dms
    dm_step = args.dm_step
    downsample = args.downsample
    nsub = args.nsub
    m = args.m
    sigma = args.sigma
    dt = args.dt
    plot_window = args.plot_window * u.s
    timeseries = args.timeseries
    spectral_average = args.spectral_average
    num_to_plot = args.num_to_plot
    from_candidate = 0
    stokes = args.stokes
    save = args.save
    convert_to_fil = args.convert_to_fil
    rfifind_dir = 'mask'
    prepsubband_dir = 'timeseries_sps'
    
    if config.has_option('DEFAULT', 'presto_image') and config['DEFAULT']['presto_image'] and os.path.isfile(config['DEFAULT']['presto_image']):
        presto_image_path = config['DEFAULT']['presto_image']
    else:
        print('Please provide an existing path to the PRESTO singularity image')
        return
        
    if filenames == 'all' and not convert_to_fil:  # if no filename is provided, search for all .fil or .fits files in the output directory
        filenames = glob.glob(args.output_dir + '/*.fil') + glob.glob(args.output_dir + '/*.fits')
    
    if filenames == 'all' and convert_to_fil:
        filenames = glob.glob(args.logfiles_dir + '/*')  # if no filename is provided, search for all files in the logfiles directory

    if args.output_dir != os.getcwd():
            os.chdir(args.output_dir)
        
    # Run rfifind
    for filename in filenames:
        filename_extension = Path(filename).suffix
        if filename_extension != '.fil' and filename_extension != '.fits' and not convert_to_fil:
            print('The file is not a filterbank nor a PSRFITS, please convert the file to a .fil or .fits file' + '\n')
            print('For ARTE logfiles you can convert them to filterbanks using the following flag: ' + '\n')
            print('--convert_to_fil' + '\n')
            print('And make sure to add the corresponding arguments for a succesful conversion, you can see them using the -h flag' + '\n')
            print('Exiting...')
            return
        
        if filename_extension != '.fil' and convert_to_fil:
            print('Converting ARTE logfile to filterbank...')
            bandpass = args.bandpass
            time_resolution = args.time_resolution
            nchannels = args.nchannels
            flip_data = args.flip_data
            padding = args.padding
            remove_samples = args.remove_samples
            logfiles_dir = Path(args.logfiles_dir)
            output_dir = Path(args.output_dir)
            full_path, start_time = converter.get_date_from_logfile(logfiles_dir, filename)
            frequencies = converter.get_frequencies(bandpass, nchannels)
            bandwidth = bandpass[0] - bandpass[1]
            path_save_fil = converter.get_output_path(output_dir, full_path)
            sigproc_object = converter.write_header_for_fil(
            path_save_fil.as_posix(),
            'ARTE',
            nchannels,
            bandwidth/nchannels,
            frequencies[0],
            time_resolution,
            start_time,
            32
        )    
            data, avg_pow,  clip, t, bases,flags, data_new,snr = converter.get_image_data_temperature([full_path.as_posix()])
            if remove_samples > 0:
                data = converter.remove_samples(data, remove_samples)

            if padding > 0:
                data = converter.add_padding(data, padding)

            if flip_data:
                data = converter.flip_data(data)
            
            data = data.astype('float' + str(32))
        
            converter.write_data_to_fil(sigproc_object, path_save_fil.as_posix(), data)
            print('Final shape:', data.shape) 
            print("File:", filename, "converted to filterbank!")
            filename = path_save_fil.name
            filename_extension = Path(filename).suffix
            # check if the conversion was succesful
            if not converter.check_conversion(path_save_fil):
                print('Conversion was not succesful, exiting...')
                return
            subprocess.run(["rm", 'readfile_output.txt'])
            print('Conversion was succesful, continuing with the FRB search pipeline...')

        filename_without_extension = Path(filename).stem
        print('Starting the FRB search pipeline, file is being processed.')
        print('Filename:', Path(filename).name + ' is being processed.')
        print('Data will be processed in the directory:', os.getcwd())
        
        #file_name = Path(filename).stem
        
        #print('Starting the FRB search pipeline, file is being processed.')
        
        start_message = 'Starting the FRB search pipeline, file is being processed\.'
        send_message(start_message)

        if args.mask:
            print('Skipping rfifind step...')
            print('Now starting prepsubband...')

        else:
            
            subprocess.run(["singularity exec --bind $PWD " + presto_image_path + " rfifind -time 2 -o output " + filename], shell=True)
            print('Done running rfifind!, now running prepsubband...')
            
        if args.prepdata:
            print('Running prepdata instead of prepsubband...')
            send_message('Running prepdata instead of prepsubband')
            dms = [start_dm + i*dm_step for i in range(num_dms)]
            for dm in dms:
                subprocess.run(['singularity exec --bind $PWD ' + presto_image_path + ' prepdata -nobary -dm ' + str(dm) + ' -downsamp ' + str(downsample) +
                               ' -mask output_rfifind.mask' + f' -o prep_output_DM{dm} ' + filename], shell=True)
            print('Done running prepdata!, now running single_pulse_search.py...')
            send_message('Done running prepdata\!, now running single\_pulse\_search\.py')    
            
    # Run prepsubband
        else:
        	subprocess.run(["singularity exec --bind $PWD " + presto_image_path + " prepsubband -nobary -lodm " + 
            	        str(start_dm) + " -dmstep " + str(dm_step) + " -numdms " + str(num_dms) + ' -downsamp ' + str(downsample) +
            	       ' -mask output_rfifind.mask -nsub '+ str(nsub) + ' -runavg -o prep_output ' + filename], shell=True)
        	print('Done running prepsubband!, now running single_pulse_search.py...')
        
        	send_message('Done running prepsubband\!, now running single\_pulse\_search\.py')
    # Run single_pulse_search.py
        subprocess.run(["singularity exec --bind $PWD " + presto_image_path + " single_pulse_search.py -m " + str(m) + ' -t ' + 
                    str(sigma) + ' -b ' + ' *.dat'], shell=True)

        print('Done running single_pulse_search.py!, now reading in candidates...')
        
        send_message('Done running single\_pulse\_search\.py\!, now reading in candidates')

    # read in the candidates from the .singlepulse files
        output = 'candidates.csv'
        output_filtered = '{}_filtered.{}'.format(*output.split('.'))
        df, df_ = cands.candidates_file(dt)
        if save:
            df.to_csv(output, index=False)
            df_.to_csv(output_filtered, index=False)
        df_.reset_index(drop=True, inplace=True)
        
        if len(df_) < 1: #no candidates
        	num_plotted = 0
        	print('No candidates where found with your parameters.')        
        elif len(df_) > num_to_plot:
            num_plotted = num_to_plot
            print('Plotting ' + str(num_to_plot) + ' candidates out of ' + str(len(df)) + ' candidates...')
            
            send_message('Plotting ' + str(num_to_plot) + ' candidates out of ' + str(len(df)) + ' candidates')
        else:
            num_plotted = len(df_)
            print('Plotting ' + str(len(df_)) + ' candidates out of ' + str(len(df)) + ' candidates...')
            
            send_message('Plotting ' + str(len(df_)) + ' candidates out of ' + str(len(df)) + ' candidates')
        
        filename_without_extension = Path(filename).stem
        os.makedirs(filename_without_extension, exist_ok=False)
        
        if len(df_) <1:
        	no_candidate_file = 'no_candidates.txt'
        	with open(no_candidate_file, 'w') as f:
        		f.write('No candidates found, sorry :(\n')
        	subprocess.run(["mv", no_candidate_file, filename_without_extension])
        else:
        	pw.plot_waterfall_from_df(Path(filename).name, df_, plot_window, downsample, filename_extension, num_to_plot, from_candidate,
                                  stokes=stokes, ts=timeseries, save=save, sed=spectral_average)
        print('Done plotting candidates!, now rearranging files...')
        
        send_message('Done plotting candidates\!, now rearranging files')
        # plot the candidates
        # move rfifind files to mask directory
        
        #os.makedirs(rfifind_dir, exist_ok=True)
        #os.makedirs(prepsubband_dir, exist_ok=True)
        #os.makedirs('candidates', exist_ok=True)
        

        # move rfifind files to mask directory
        '''
        subprocess.run(['mv', 'output_rfifind.mask', rfifind_dir])
        subprocess.run(['mv', 'output_rfifind.bytemask', rfifind_dir])
        subprocess.run(['mv', 'output_rfifind.inf', rfifind_dir])
        subprocess.run(['mv', 'output_rfifind.ps', rfifind_dir])
        subprocess.run(['mv', 'output_rfifind.rfi', rfifind_dir])
        subprocess.run(['mv', 'output_rfifind.stats', rfifind_dir])
        '''
        
        os.remove('output_rfifind.mask')
        os.remove('output_rfifind.bytemask',)
        os.remove('output_rfifind.rfi')
        os.remove('output_rfifind.stats')
        subprocess.run(['mv', 'output_rfifind.ps', filename_without_extension])
        
        # move prepsubband files to timeseries_sps directory
        
        
        subprocess.run(["rm *.dat "], shell=True)
        subprocess.run(["rm *.inf "], shell=True)
        subprocess.run(["rm *.singlepulse "], shell=True)
        #subprocess.run(["/bin/bash", "-i", "-c", "source ~/.bashrc; mv *.dat " + prepsubband_dir])
        #subprocess.run(["/bin/bash", "-i", "-c", "source ~/.bashrc; mv *.inf " + prepsubband_dir])
        #subprocess.run(["/bin/bash", "-i", "-c", "source ~/.bashrc; mv *.singlepulse " + prepsubband_dir])

        subprocess.run(["mv", "prep_output_singlepulse.ps", filename_without_extension])
        # move candidates tables and pdf to candidates directory
        subprocess.run(["mv *.csv " + filename_without_extension], shell=True)
        
        if not len(df_)<1:
        	subprocess.run(["mv *.pdf " + filename_without_extension], shell=True)
        # move all files to a directory with the name of the .fil file
        '''
        subprocess.run(["mv", rfifind_dir, filename_without_extension])
        subprocess.run(["mv", prepsubband_dir, filename_without_extension])
        subprocess.run(["mv", "candidates", filename_without_extension])
        '''
        
        # make summary file
        summary_file = filename_without_extension + '_summary.txt'
        write_summary(summary_file, filename_without_extension, df, df_, num_plotted, downsample, nsub, m, sigma,
                     dt, plot_window, timeseries, spectral_average, start_dm, num_dms, dm_step,
                        num_to_plot, stokes)

        subprocess.run(["mv", summary_file, filename_without_extension])
        print('Summary file created: ' + summary_file)
        
        
        if args.remove_fil: #and convert_to_fil:
            subprocess.run(["rm", filename])
            print('Removed ' + filename)
        
        
        
        print(Path(filename).name + ' Done!, Hope you found some FRBs/Pulsars!')
        end_message = 'Done\!, Hope you found some FRBs/Pulsars\!'
        send_message(end_message)
        del(filename)
        
    if args.wrap_pdfs:
            pdf_files = collect_pdfs(os.getcwd())
            merge_pdfs(pdf_files, 'all_candidates.pdf')
            delete_pdfs(pdf_files)
            print('All pdfs wrapped into all_candidates.pdf')
if __name__ == '__main__':
    main()

    

    






    



