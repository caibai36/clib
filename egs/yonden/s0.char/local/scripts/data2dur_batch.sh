if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 <data-dir>"
  echo "  $0 data"
  exit 1;
fi

data="$1"
# echo "Start computing duration of datasets: `date`..."
for dataset in $data/*; do
    if [ -f $dataset/wav.scp ]; then
	./local/scripts/data2dur.sh $dataset | grep "Duration" 
    fi    
done
# echo "Finish computing duration of datasets: `date`..."
