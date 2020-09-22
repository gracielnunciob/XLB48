import os 傻

def binary_search(srch_list,key):
	lo = 0
	hi = len(srch_list)
	while lo < hi:
		m = lo + (hi - lo) // 2
		if srch_list[m] == key:
			return m
		elif srch_list[m] > key:
			hi = m
		else:
			lo = m + 1
	return -1

def norm(song):
	song = song.strip()
	if len(song) < 4 or song[-4:] != ".wav":
		song += ".wav"
	return song

# list of all the songs in the folder
song_files = []
root_list = []
for root, dirs, files in os.walk(".", topdown=False):
	for name in files:
		if '.wav' in name:
			root_list.append(root.strip())
			song_files.append(name.strip())

song_files = sorted(song_files)
print(len(song_files))

# for f in song_files:
	# print(f)

songs = []
with open('Song List.txt', 'r') as f:
	for line in f:		
		# our playlist of 419 songs
		songs.append(norm(line))

	print(len(songs))

songs = sorted(songs)

with open("Song List.txt","w") as f:
	f.write("\n".join(list(map(norm,songs))))

songs = sorted(songs)

# for song in song_files:
# 	print("Song File: |{}|".format(song))
total_ctr = 0
num_in_list = 0
num_missing = 0

no_match = []
match = []

for song in songs:
	total_ctr += 1
	if binary_search(song_files,song) != -1:
		num_in_list += 1
		match.append(song)
	else:
		num_missing += 1
		no_match.append(song)

no_match = sorted(no_match)

print("Total Song Files: {}".format(total_ctr))
print("Total In List: {}".format(num_in_list))
print("Total Not In List: {}".format(num_missing))
# print(os.path.join(root_list[0], song_files[0]))

# for song in match:
# 	print("Matched Song File: {}".format(song))

# for song in no_match:
# 	print("Missing Song File: |{}|".format(song))

# os.remove(os.path.join(root_list[0], song_files[0]))