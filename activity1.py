import pandas as pd
import numpy as np

def main():
    mutations = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
    num_of_people = mutations.shape[0]
    num_of_mutations = mutations.shape[1]
    print('Number of unique mutations: ', num_of_mutations)
    print('Number of individual samples: ', num_of_people)

    print('Number of unique mutations for patient C1: ', mutations.loc['C1', :].agg(sum))
    print('Number of unique mutations for patient NC1: ', mutations.loc['NC1', :].agg(sum))

    avg_mutations = 0
    min_max_mutations = []
    for i in mutations.index:
        agg = mutations.loc[i, :].agg(sum)
        avg_mutations += agg
        min_max_mutations.append(agg)
    avg_mutations = avg_mutations / num_of_people
    print('Average number of mutations per patient: ', avg_mutations)
    print('Minimum number of mutations per patient: ', min(min_max_mutations))
    print('Maximum number of mutations per patient: ', max(min_max_mutations))

    # get the number of mutations for BRAF and KRAS genes
    # should avoid overlap for good form, despite only being one column
    braf_genes = []
    for i in mutations.columns:
        if 'BRAF' in i:
            braf_genes.append(i)
    patients_with_braf_mut = sum(mutations.loc[:, braf_genes].agg(sum))
    print('Number of patients who have mutations in BRAF genes: ', patients_with_braf_mut)

    # needs to avoid overlap
    kras_genes = []
    for i in mutations.columns:
        if 'KRAS' in i:
            kras_genes.append(i)
    kras_list = np.array(mutations.loc[:, kras_genes])
    kras_list_binary = []
    for i in kras_list:
        list(map(lambda x: kras_list_binary.append(x), \
                 list(map(lambda x: 1 if x >= 1 else 0, [sum(i)]))))

    patients_with_kras_mut = sum(kras_list_binary)
    print('Number of patients who have mutations in KRAS genes: ', patients_with_kras_mut)

    # get the average number of patients for each mutation
    avg_patients_per_mutation = 0
    min_max_patients = []
    for i in mutations.columns:
        agg = mutations.loc[:, i].agg(sum)
        avg_patients_per_mutation += agg
        min_max_patients.append(agg)

    avg_patients_per_mutation = avg_patients_per_mutation / num_of_mutations
    print('Average number of patients per mutation: ', avg_patients_per_mutation)
    print('Minimum patients per mutation: ', min(min_max_patients))
    print('Maximum patients per mutation: ', max(min_max_patients))
    
    
        


if __name__ == '__main__':
    main()
