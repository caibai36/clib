if [ $# != 1 ]; then
  echo "Usage: "
  echo "  ./data2dur.sh <data-dir>"
  echo "  ./data2dur.sh data/train"
  exit 1;
fi

dataset="$1"
echo "Start computing duration of dataset: `date`..."
utils/data/get_utt2dur.sh ${dataset}
echo -ne "Duration: $1\t" 
#cat ${dataset}/utt2dur | perl -ne '@line=split; $s+=$line[1]; END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %dhour:%dmin:%.1fsec",$s, $h,$m, $r;}'
cat ${dataset}/utt2dur | perl -ne '@line=split; $s+=$line[1]; END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%dh%dm\n", $h,$m;}'
echo "Finish computing duration of dataset: `date`..."
