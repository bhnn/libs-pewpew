import csv
import platform
import time
from random import sample, uniform

import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

project_path = './ml'

def read_mineral_list(file_path, allow_pickle=True):
    """
    Reads list of all minerals, checks them for possible mistakes, reformats into easily iterable list
    """
    print('BEGIN Read list of all minerals...')
    minerals = np.load(file_path, allow_pickle=allow_pickle)
    print('DONE! Reading minerals.')
    return minerals

def fix_sum_to_100(composition, method=0):
    """
    Fixes sum of passed atomic composition to sum up to 100.0 
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
    Assembles NIST LIBS link from elements and atomic composition.
    """
    print('  BEGIN Assemble NIST LIBS URL...')
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
    print('  DONE! Assembling NIST LIBS URL.')
    return request_url

def get_webdriver():
    """
    Construct Chrome webdriver for Selenium.
    """
    print('BEGIN Get Selenum webdriver...')
    chrome_options = Options()
    # headless browser, no gui
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-extentions')

    driver_path = 'lib/chromedrivers/cd'
    
    if platform.system() == 'Linux':
        driver_path += '_linux'
    elif platform.system() == 'Darwin':
        driver_path += '_osx'
    elif platform.system() == 'Windows':
        driver_path += '_win.exe'
    else:
        raise ValueError('Chromedriver encountered invalid platform information!')

    driver = webdriver.Chrome(driver_path, options=chrome_options)
    print('DONE! Getting Selenium webdriver.')
    return driver

def retrieve_nist_data(driver, link, resolution=1000):
    """
    Open NIST LIBS link, recalculate resolution to specified value, extract spectral information.
    """
    print('  BEGIN Retrieve NIST data using Selenium...')
    # open generated link in browser
    driver.get(link)

    # wait for spectrum to load and the resolution textbox to unlock
    res_input = WebDriverWait(driver, 240).until(
        EC.presence_of_element_located((By.ID, 'resolution'))
    )

    # remove all content from it, then enter desired resolution and submit
    res_input.send_keys(Keys.CONTROL + 'a')
    res_input.send_keys(Keys.DELETE)
    res_input.send_keys(resolution)
    res_input.send_keys(Keys.ENTER)

    # wait until calculation is complete, then click Download CSV button
    csv_btn = WebDriverWait(driver, 240).until(
        EC.presence_of_element_located((By.NAME, 'ViewDataCSV'))
    )
    csv_btn.click()

    # switch to new tab with CSV data (js generates it with fixed name), then grab all text and close
    driver.switch_to.window('libs_data')
    text = BeautifulSoup(driver.page_source, 'lxml').text
    start = text.find('Wavelength')
    text = text[start:]
    
    print('  DONE! Retriebing NIST data.')
    return text

def clean_nist_data(data, use_header=False):
    """
    Removes unnecessary spectral information and reformats from string to list.
    """
    print('  BEGIN Clean and format NIST data...')
    # data is read as string, has to be converted into list of lists
    # only keep first 2 columns (wavelength + sum intensity)
    split_data = [line.split(',')[:2] for line in data.split('\n')[int(not use_header):]]
    # clean up invalid data
    cleaned_split_data = [l for l in split_data if len(l) == 2]
    # convert from string to float to story in numpy array later
    float_data = np.array(cleaned_split_data).astype(float)
    print('  DONE! Cleaning formatting NIST data.')
    return float_data

def write_to_file(data, filename):
    """
    Write collected information to file.
    """
    print('  BEGIN Write data to npy file...')
    np.save(filename, data)
    print('  DONE! Writing data to npy file.')

def reset_webdriver(driver):
    """
    Resets Selenium webdriver between elements.
    """
    # clear out cookies -- not sure if needed
    driver.delete_all_cookies()

    # get list of current tabs
    current_tab = driver.current_window_handle
    tab_list = driver.window_handles
    tab_list.remove(current_tab)
    
    # close all but one open tabs from last session
    for tab in tab_list:
        driver.switch_to.window(tab)
        driver.close()
    
    driver.switch_to.window(current_tab)
    driver.get('about:blank')
    return driver

def calculate_random_segments(num_segments):
    """
    Splits the value ranges of electron temperature and density into several segments to allow controlled randomisation
    of the entire domain.
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

def create_synthetic_dataset(data_source_path, measurement_segmentation, amount_variations):
    """
    Main routine to generate synthetic dataset.
    """
    print('BEGIN Create synthetic mineral dataset...')
    # read in list of all elements
    all_minerals = read_mineral_list(data_source_path)

    # webdriver for NIST data access, persistent outside of loop to cut down load times of instance creation
    driver = get_webdriver()
    for num, name, elements, composition, m_class, m_group in all_minerals:
        print(f'BEGIN Next element: {name}')
        # calculate random alterations of electron temperature and density (measurement noise)
        for i,(t_ev,e_den) in enumerate(calculate_random_segments(measurement_segmentation)):
            # calculate random alterations of composition
            print(f'  BEGIN Draw from randomisation segment {i+1} of {measurement_segmentation-1}')
            variations = calculate_variations(composition, amount_variations)
            for j,var in enumerate(variations):
                print(f'  BEGIN Atomic variation {j+1} of {len(variations)}')
                # make sum of composing elements be exactly 100
                fix_sum_to_100(var, method=0)

                # create NIST link, collect data
                t_ev_sample = uniform(t_ev[0], t_ev[1])
                e_den_sample = str(uniform(e_den[0], e_den[1])).replace('+','') # NIST site malfunctions for + in exp notation
                link = assemble_nist_url(elements, var, resolution=1000, temp=t_ev_sample, eden=e_den_sample)
                # clean up results, discard columns
                data = retrieve_nist_data(driver, link, 1000)
                data = clean_nist_data(data)
                # save to file
                data = np.array([data, np.array([m_class, m_group, num])])
                
                # <laufende nummer>_<klasse>_<gruppe>_<ausführung> 
                new_file_name = f'{num:04}_{m_class:03}_{m_group:03}_{i:03}_{j:05}'
                write_to_file(data, f'results/{new_file_name}')

                # reset webdriver for next iteration
                driver = reset_webdriver(driver)
                print(f'  DONE! Atomatic variation {j+1} of {len(variations)}')
                print(f'  DONE! Drawing from randomisation segment {i+1} of {measurement_segmentation-1}')
                print(f'DONE! with element {name}')
    # clean up
    driver.quit()

    print('DONE! Creating synthetic mineral dataset.')

# suppress early interruption stacktrace for convenience
try:
    start_time = time.time()
    create_synthetic_dataset('data/synthetic_minerals.npy', 6, 1000)
except KeyboardInterrupt as e:
    pass

print('Execution interrupted.')
print(f'Runtime: {time.time() - start_time:.2f} seconds')

# todo
# - fix_sum_to_100 before/after calculating variations?
# - variations: remove elements smaller than 4% with 1:15 chance

# test https://physics.nist.gov/cgi-bin/ASD/lines1.pl?composition=H%3A28.13%3BC%3A0.78%3BNa%3A1.17%3BMg%3A1.48%3BAl%3A3.67%3BSi%3A9.38%3BCa%3A3.52%3BFe%3A2.66%3BO%3A49.209999999999994&mytext%5B%5D=H&myperc%5B%5D=28.13&spectra=H0-2%2CC0-2%2CNa0-2%2CMg0-2%2CAl0-2%2CSi0-2%2CCa0-2%2CFe0-2%2CO0-2&mytext%5B%5D=C&myperc%5B%5D=0.78&mytext%5B%5D=Na&myperc%5B%5D=1.17&mytext%5B%5D=Mg&myperc%5B%5D=1.48&mytext%5B%5D=Al&myperc%5B%5D=3.67&mytext%5B%5D=Si&myperc%5B%5D=9.38&mytext%5B%5D=Ca&myperc%5B%5D=3.52&mytext%5B%5D=Fe&myperc%5B%5D=2.66&mytext%5B%5D=O&myperc%5B%5D=49.209999999999994&low_w=180&limits_type=0&upp_w=960&show_av=2&unit=1&resolution=1000&temp=1&eden=1e17&libs=1
