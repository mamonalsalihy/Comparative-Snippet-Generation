docker build -t feng-hirst .
for file in ./tmp/txt/*
do
  file_path="$(echo $file | cut -d'.' -f2-)"
  docker run -v /mnt/d/Brain/Comparative_Snippet_Generation/feng-hirst-rst-parser/tmp:/tmp -ti feng-hirst $file_path --skip_parsing
done