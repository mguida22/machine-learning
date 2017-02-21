# Get name data
if [ -f "first_names.txt" ]; then
	echo "already have first_names.txt"
else
  wget -O first_names.txt http://deron.meranda.us/data/census-derived-all-first.txt
	echo "downloaded first_names.txt"
fi

if [ -f "last_names.txt" ]; then
	echo "already have last_names.txt"
else
  wget -O last_names.txt http://www2.census.gov/topics/genealogy/1990surnames/dist.all.last
	echo "downloaded last_names.txt"
fi

if [ -f "common_words.txt" ]; then
  echo "already have common_words.txt"
else
  wget -O common_words.txt https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt
  echo "downloaded common_words.txt"
fi

if [ -f "names.txt" ]; then
  echo "already have names.txt"
else
  # take only first column (names) and lowercase it
  cat first_names.txt last_names.txt | sed 's/[^A-Z]//g' | tr "[:upper:]" "[:lower:]" >> names.txt
  # remove dictionary words from the list
  awk 'NR==FNR{a[$0];next} !($0 in a)' common_words.txt names.txt > tmp.txt && mv tmp.txt names.txt
  echo "generated names.txt"
fi
