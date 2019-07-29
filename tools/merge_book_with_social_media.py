#
# Appending the hammaburg_book in form of social messages to existing csv file
#
input_file = "data/hammaburg_book.txt"

output_file = "data/database_lndmhh_hammaburg_2018.csv"

date = "2018-05-11 00:01:00"

f = open(input_file, 'r+', encoding='utf8')
data = f.readlines()

output = open(output_file, 'a', encoding='utf8')

for line in data:
    newLine = date+ "; hammaburg_book; hamburg ; ;\""+ line.rstrip('\n') + "\"; fake"
    output.write(newLine + '\n')
output.close()