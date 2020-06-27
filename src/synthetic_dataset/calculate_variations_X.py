from random import uniform, sample

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

    # 2)
    #   roll random 0.0 - 3.0 (relative) to take from all elements in a row, randomised
    #   when sum of 3% taken is reached, roll random 0.0 - 3.0 (relative) to give back 
    #   to all elements in a row, randomised
    #   check that small elements dont get more than 3% (relative)

comp = [28.13, 0.78, 49.22, 1.17, 1.48, 3.67, 9.38, 3.52, 2.66]
variations = calculate_variations(comp, 10)
print('{:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18}'.format(*comp))
for var in variations:
    diffs = list()
    for j in range(len(var)):
        diffs.append(abs(var[j] / comp[j]))
    print('{:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18} {:<18}'.format(*var))
    print('{:<18.4} {:<18.4} {:<18.4} {:<18.4} {:<18.4} {:<18.4} {:<18.4} {:<18.4} {:<18.4}'.format(*diffs),'\n')