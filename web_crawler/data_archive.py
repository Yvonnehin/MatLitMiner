from pandas import DataFrame

def data_archive(sample,filename,sheet):   
    data  = DataFrame(sample)
    DataFrame(data).to_excel(filename,sheet_name=sheet)
    

if __name__ == '__main__':
    sample = [[1,2],[1,3],[3,4]]
    data_archive(sample,"1.xlsx","test")