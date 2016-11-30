#!/usr/bin/python

import os
import zipfile
import re
import sys


pattern = re.compile('[\W_]+', re.UNICODE)


def file_cleaner(filename):
    with open(filename, 'r') as f:
        for line in f:
            data = line.decode('utf-8').strip().split("\t")
            main_article = data[1].lower()
            main_article = re.sub(pattern, ' ', main_article)
            main_article = main_article.encode('utf-8', errors='ignore')
            with open("converted", 'a+') as converted_file:
                converted_file.write("{}\n".format(main_article))


def clean_and_zip(argument):
    if not os.path.isdir(argument):
        file_cleaner(argument)
    else:
        list_of_files = os.listdir(argument)
        print list_of_files
        for individual_file in list_of_files:
            file_cleaner(argument + "/" + individual_file)

    with zipfile.ZipFile("converted.zip", "w") as the_file:
        the_file.write("converted")

    os.remove("converted")

if __name__ == "__main__":
    clean_and_zip(sys.argv[1])
