#!usr/bin/bash

# This file is useful to generate data for the memory app. Formatting the database
# as a SQLLite database is also possible. To that end, run the following command
# in the terminal from the root directory of the project.
# ```shell
# bash <path_to_file_folder>/generate_data.sh
# ```
#

python -m apps.memory.dataset.generate people
python -m apps.memory.dataset.generate biographies --num 1000
python -m apps.memory.dataset.generate qa --num 100
python -m apps.memory.dataset.generate qa --tooluse --num 100

printf 'Do you want to format the database as a SQLlite database? (Y/N)? '
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "SQLlite formatting."
    python -m apps.memory.dataset.database create
fi

echo "Data generated successfully!"