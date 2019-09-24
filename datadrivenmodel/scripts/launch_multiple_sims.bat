set brain=%1
for /l %%x in (1, 1, 10) do (
	echo %%x
	start python hub.py --brain %brain%
	timeout 5
	)