#!usr/bin/bash

# This file is useful to generate OOD data for the memory app.
# To that end, run the following command in the terminal from the root directory of the project.
# ```shell
# bash <path_to_file_folder>/generate_ood_data.sh
# ```

echo "Generating OOD data for the memory app..."
echo "...people"
python -m apps.memory.dataset.ood_data.generate people
echo "...biographies"
python -m apps.memory.dataset.ood_data.generate biographies ${1:+--num $1}
echo "...question/answer"
python -m apps.memory.dataset.ood_data.generate qa ${1:+--num $1}
echo "...question/answer with tool use"
python -m apps.memory.dataset.ood_data.generate qa --tooluse ${1:+--num $1}

printf 'Do you want to format the OOD dataset as a SQLlite database? (Y/N)? '
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "SQLlite formatting."
    python -m apps.memory.dataset.ood_data.database delete
    python -m apps.memory.dataset.ood_data.database create
fi

echo "Data generated successfully!"