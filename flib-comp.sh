current=$(pwd)
source=${current}/flib_java/
target=${current}/flib_class/

cd ${source}
make

cd ${target}
files=$(find * -type f -name "*.class")
jar -cf filter-library.jar $files
jar -i filter-library.jar
cp filter-library.jar ${current}/jars/forest-library.jar

cd ${current}
