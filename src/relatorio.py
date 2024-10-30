import csv


def table_to_csv(query_result, out_path, cols):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(cols)
        writer.writerows(query_result)
