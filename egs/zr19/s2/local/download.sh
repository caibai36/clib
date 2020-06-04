mkdir -p db
wget https://download.zerospeech.com/2019/english.tgz
tar xvfz english.tgz -C db
rm -f english.tgz

wget https://download.zerospeech.com/2019/english_small.tgz
tar xvfz english_small.tgz -C db
rm -f english_small.tgz

wget https://download.zerospeech.com/2019/surprise.zip
unzip -P 9kneopShevtat] surprise.zip -d db
# enter the password when prompted for
rm -f surprise.zip
