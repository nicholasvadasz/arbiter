### SQL Query Example ###
# SQLer
import csv
import pandas

def SQLer(file_pt, nrows=10):
    table = pandas.read_csv(file_pt, sep='\t', nrows=nrows,
                 engine='python')
    f_dict = table.to_dict()
    key = list(f_dict.keys())[0]
    query = f_dict[key]

    columns = '|'.join(key.split(','))

    test_column = key.split(',')[0]
    print(table)
    print("Here \n")
    print(table[table.keys()[0]][0])
    
    print("\n")
    print(f'colmn_names: {columns}')
    print(query)



### Sample script
file_pt = 'finance_sheet.csv'
SQLer(file_pt)