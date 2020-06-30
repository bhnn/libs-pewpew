import argparse
import multiprocessing as mp
import sys
import time
from os import getpid, makedirs
from os.path import join, exists
from random import sample, uniform

import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup


def read_mineral_list(file_path, allow_pickle=True):
    """
    Reads list of all minerals, checks them for possible mistakes, reformats into easily iterable list

    :param file_path: Path to file containing mineral information
    :param allow_pickle: Argument to pass to numpy.load()'s allow_pickle parameter. Defaults to true, current list is pickled.
    :returns: List of mineral information (id, name, compositions, class, group, ...)
    """
    minerals = np.load(file_path, allow_pickle=allow_pickle)
    return minerals

def fix_sum_to_100(composition, method=0):
    """
    Fixes sum of passed atomic composition to sum up to 100.0. The NIST website that generates LIBS spectra requires a
    sum of even 100.0 for all its inputs. Because of notation inaccurates on the Mineralienatlas website, spectra with a
    sum of 99.<1-9> or 100.<1-9> are possible, thus the need to adjust the sum.

    :param composition: List containing atomic composition of single mineral
    :param method: Picks the distribution method used to fix the compositional sum to 100. Method 0 distributes difference 
    to 100 evenly, method 1 adjusts the largest element in the composition to reach 100.0. Defaults to 0 because even
    distribution delivers more chemically sound results.
    :returns: List containing atomic composition of single mineral with a sum of 100.0
    """
    if method == 0:
        # distribute difference of sum to 100.0 evenly across all composing elements
        if sum(composition) != 100:
            diff = (sum(composition) - 100) / len(composition)
            for i in range(len(composition)-1):
                composition[i] -= diff
            composition[-1] -= sum(composition) - 100
    elif method == 1:
        # subtract difference of sum to 100.0 from largest composing element
        if sum(composition) != 100:
            index = composition.index(max(composition))
            composition[index] -= sum(composition) - 100
    return composition

def assemble_nist_url(elements, percentages, low_wl=180, high_wl=960, resolution=1000, temp='1', eden='1e17'):
    """
    Assembles NIST LIBS url from elements and atomic composition.

    :param elements: List of names of compositional elements (str)
    :param percentages: List/ndarray of composition percentages in the same order as the names of the corresponding elements
    :param low_wl: Lower bound of wavelengths produced in generation process
    :param high_wl: Upper bound of wavelengths produced in generation process
    :param resolution: Resolution to be used in generation process. Rarely works because the website downscales minerals
    whose composition is more complex than 2 minerals to resolution 100-200
    :param temp: Electron temperature measurement parameter used in generation process
    :param eden: Electron density measurement parameter used in generation process. Needs to be passed as a string, such as
    '1e17'. Passing the same value numerically will result in the addition of a + when adding it to the URL (1e+17),
    causing the website to malfunction.
    :returns: URL string to request synthetic spectrum of specified mineral from the NIST website
    """
    assert type(elements) is list
    assert type(percentages) is list or np.ndarray
    content_len = len(elements)
    # base URL
    request_url = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl'
    # encoded composition string of all composing elements: El1:Perc1;El2:Perc2;...
    comp_string = '?composition=' + '%3B'.join([f'{elements[i]}%3A{percentages[i]}' for i in range(content_len)])
    # encoded spectra string for all composing elements: El10-2,El20-2,El30-2,...
    spec_string = '&spectra=' + '%2C'.join([f'{elements[i]}0-2' for i in range(content_len)])

    for i in range(content_len):
        # add comp_string and spec_string only for first element
        request_url += '{}&mytext%5B%5D={}&myperc%5B%5D={}{}'.format(comp_string if i == 0 else '', 
                                                                     elements[i], 
                                                                     percentages[i], 
                                                                     spec_string if i == 0 else '')

    request_url += f'&low_w={low_wl}&limits_type=0&upp_w={high_wl}&show_av=2&unit=1&resolution={resolution}'
    request_url += f'&temp={temp}&eden={eden}&libs=1'
    return request_url

def download_text_and_header(url, verify=False, timeout=1):
    """
    Read page source from url and split into body and header.

    :param url: URL to request
    :param verify: Request.get()'s verify parameter, defaults to False
    :param timeout: Request.get()'s timeout parameter, defaults to 1
    :returns: Tuple containing the entire content of the url's (body, header)
    """
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'}
    response = requests.get(url, verify=verify, timeout=timeout, headers=headers)
    html = response.text
    header = response.headers
    return (str(html),str(header))

def retrieve_nist_data(url):
    """
    Download script body information from NIST LIBS url, split off stick data and separate into wavelengths and intensities.

    :param url: NIST website url used to generate spectra
    :returns: Tuple of stick plot data (infinite resolution raw data) of the generated (stick_wavelength, stick_intensity)
    """
    (body, _) = download_text_and_header(url, True, timeout=30)
    soup = BeautifulSoup(body, "lxml")

    rawJ = soup.find_all('script')
    J = str(rawJ[5])
    J1 = J.split(' var dataSticksArray=')
    J2 = J1[1].split(']];')
    var_array = J2[0] + ']];'
    lines = var_array.split('\n')

    wave_length = []
    sums = []
    for i in range(len(lines)-1):
        cur_line = lines[i+1].replace('[', '').replace('],', '')
        split_line = cur_line.split(',')
        cur_wl = np.float(split_line[0])
        cur_sum = 0
        for j in range(len(split_line)-1):
            try:
                cur_val = np.float(split_line[j+1])
            except:
                cur_val = 0.0
            cur_sum += cur_val
        wave_length.append(cur_wl)
        sums.append(cur_sum)

    stick_wavelengths = np.array(wave_length)
    stick_intensities = np.array(sums)
    
    return (stick_wavelengths, stick_intensities)

def build_spectral_data(data, FWHM):
    """
    Build spectral peaks from stick plot data and interpolates everything to the PhysChem wavelength defaults. 
    Function provided by Federico Malerba.

    :param data: Tuple of stick plot data (infinite resolution raw data) of the generated (stick_wavelength, stick_intensity)
    :param FWHM: "Full width at half maximum" parameter for calculating Gaussian curves
    :returns: ndarray containing generated, interpolated spectrum (wavelength, intensity) of generated mineral
    """
    stick_wl, stick_int = data

    new_wavelengths = np.round(np.arange(180, 961, 0.1), 1)
    n = new_wavelengths.size
    m = stick_wl.size

    standard_deviation = FWHM / 2.3548200450309493
    # parameter coming from mathematical theory

    # we want to determine all the intensities in one go using efficient matrix computations, so we
    # construct a matrix with n. rows = len(new_wavelengths) and n. cols = len(wavelengths)
    distances = np.reshape(np.repeat(new_wavelengths, m), [n, m]) - stick_wl
    new_intensities = np.sum((stick_int * np.exp(-np.square(distances) / (2*np.square(standard_deviation)))), axis=1)

    return np.stack((new_wavelengths, new_intensities), axis=1)

def write_to_file(data, filename):
    """
    Write collected information to file.

    :param data: Data to include in file
    :param filename: Filename for serialisation
    """
    np.save(filename, data)

def calculate_random_segments(num_segments):
    """
    Splits the value ranges of electron temperature and density into several segments to allow controlled randomisation
    of the entire domain.

    :param num_segments: Number of segments (-1) to split the electron temperature/density intervals into
    :returns: List of tuples containing the lower and upper bound of each segment
    """
    # charge and density boundaries computed from real-world measurement averages sampled from physical chemistry dept.
    t_ev_segments = np.linspace(0.73, 1.12, num=num_segments)
    t_ev_ranges = [(t_ev_segments[i], t_ev_segments[i+1]) for i in range(num_segments-1)]
    eden_segments = np.linspace(5.5e+16, 1.97e+17, num=num_segments)
    eden_ranges = [(eden_segments[i], eden_segments[i+1]) for i in range(num_segments-1)]
    return zip(t_ev_ranges, eden_ranges)

def calculate_variations(source, quantity, min_noise=1e-16, max_noise=0.05):
    """
    Calculates quantity many alterations of the passed atomic composition.
    Formula:
        - start with max_noise worth of possible variation
        - for each element, random roll between [1e-16, max_noise) of variation to subtract from it
        - decrease max_noise by the result of each roll, collect subtracted amount
        - for each element, roll between [0, sum_sum_removed) to add to the element

    :param source: Source composition that serves as base for any variations
    :param quantity: Amount of different variations to produce
    :param min_noise: Minimal amount of noise to include during the first step of randomisation, 1e-16 in place of small non-0
    :param max_noise: Maximum amount of noise to work with in the randomisation process. 5% (0.05) was deemed a realistic
    value by the Department of Physical Chemistry
    :returns: List containing quantity many variations of the source elemental composition passed to the function.
    """
    results = list()
    for _ in range(quantity):
        # duplicate source list
        copy = source[:]
        top_noise = max_noise
        sum_removed = 0
        # iterate over shuffled indices of original list, treats list as scrambled while maintaining original order for 
        # re-use with indentically ordered original dataset values later
        for i in sample(range(len(source)), len(source)):
            # determine noise intensity, reduce top_noise
            noise = uniform(min_noise, top_noise)
            top_noise -= noise

            # subtract calculated noise from current element, noise: [1e-16, 0.05-)
            take_amount = copy[i] * noise
            copy[i] -= take_amount
            sum_removed += take_amount
        
        give_range = list(range(len(source)))
        while(sum_removed != 0 and give_range):
            for i in sample(give_range, len(give_range)):
                # determine amount of noise to give back to current element
                give_amount = uniform(0, sum_removed)
                # maximum additional mass the current element can legally receive
                cap = copy[i] * 0.05
                # if amount of noise larger than element's cap, only add noise up to cap and remove element from rotation
                # this is done to make sure you don't overcap smaller elemental components, results in distributing the 
                # remaining noise among the larger elements until completely distributed. (So no further fix-to-100%
                # approaches have to be applied that would mess with the achieved distribution)
                if give_amount > cap:
                    copy[i] += cap
                    sum_removed -= cap
                    give_range.remove(i)
                else:
                    copy[i] += give_amount
                    sum_removed -= give_amount
        results.append(copy)
    return results

def create_synthetic_dataset(minerals, worker_id, dest_path):
    """
    Main routine to generate synthetic dataset.

    :param minerals: List of minerals to continually iterate over. After exhausting the list, the process repeats.
    :param worker_id: Worker id assigned to this process, passed as command line argument to ensure unique names for the
    generated files
    """
    # unique process identifier
    process_id = getpid()
    count = 0
    # Process is meant to generate spectra for all passed minerals continually until interrupted
    while(True):
        try:
            for num, name, elements, composition, m_class, m_group in minerals:
                # calculate random alterations of electron temperature and density (measurement noise)
                for i,(t_ev,e_den) in enumerate(calculate_random_segments(num_segments=6)):
                    # calculate random alterations of composition
                    num_variations = 5
                    variations = calculate_variations(composition, quantity=num_variations)
                    for j,var in enumerate(variations):
                        # make sum of composing elements be exactly 100
                        fix_sum_to_100(var, method=0)

                        # create NIST link, collect data
                        t_ev_sample = uniform(t_ev[0], t_ev[1])
                        e_den_sample = str(uniform(e_den[0], e_den[1])).replace('+','') # NIST site malfunctions for + in exp notation
                        url = assemble_nist_url(elements, var, resolution=1000, temp=t_ev_sample, eden=e_den_sample)
                        # clean up results, discard columns
                        data = retrieve_nist_data(url)
                        data = build_spectral_data(data, FWHM=0.18039)
                        # save to file
                        data = np.array([data, np.array([m_class, m_group, num])])
                        
                        # <manual worker id>_<process id>__<mineral id>_<class>_<group>_<tev/eden random segment>_<mineral count> 
                        min_num = count + j
                        new_file_name = f'{worker_id:02}_{process_id:08}__{num:04}_{m_class:03}_{m_group:03}_{i:03}_{min_num:05}'
                        print(new_file_name)
                        write_to_file(data, join(dest_path, new_file_name))
                count += num_variations
        except KeyboardInterrupt as e:
            print('KeyboardInterrupt received. Aborting.')
            break
    return 0

def start_multiprocessing(**args):
    """
    Starts the main generation routine with as many processes as specified
    """

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            repo_path = dataset_configs['repo_path']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    dest_path = join(repo_path, 'results', 'synthetic')
    makedirs(dest_path, exist_ok=True)

    # read in list of all minerals 
    mineral_subset = [mineral for mineral in read_mineral_list(join(repo_path, 'data', 'synthetic_minerals.npy'))
                      if mineral[0] in [11, 19, 26, 28, 35, 41, 73, 80, 86, 88, 97, 98]]
    
    process_args = [(mineral_subset, args['id'], dest_path)] * args['num_processes']

    with mp.Pool(args['num_processes']) as pool:
        pool.starmap(create_synthetic_dataset, process_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--processes',
        type=int,
        required=True,
        help='Amount of processes to start in parallel',
        dest='num_processes'
    )
    parser.add_argument(
        '-i', '--id',
        type=int,
        required=True,
        help='Unique process id, used for distinguishing save files by process later',
        dest='id')
    args = parser.parse_args()

    try:
        start_time = time.time()
        start_multiprocessing(**vars(args))
    except KeyboardInterrupt as e:
        # suppress early interruption stacktrace for convenience
        print('Execution interrupted.')

    print(f'Runtime: {time.time() - start_time:.2f} seconds')
