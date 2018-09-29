# From current dir, convert all pdf to png

for f in $(basename -s '.pdf' *.pdf)
do
	if test $f.pdf -nt $f.png 
	then
		echo "$f.pdf -> $f.png"
		convert -density 450x450 $f.pdf $f.png
	fi
done
