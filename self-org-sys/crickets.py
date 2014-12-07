N = 10 #crickets
M = 20 #parasites

z = 10
cvi_c = 11
cvi_p = 2
val_c = 3
val_p = 5
xc = 10
yc = 80
xp = 10
yp = 80
nc = 20
max_steps = 1000


from random import randint

par = [cvi_p] * M
cri = [cvi_c] * N
par_life = [cvi_p] * M
cri_life = [cvi_c] * N
par_last = [0] * M #years when the parasite last met a cricket

par_dead = 0
cri_dead = 0

def pass_year():
	par_dead = 0
	cri_dead = 0
	for i in range(len(par)):
		par[i] -= 1
		if par[i] == 0:
			par_dead += 1
	for i in range(len(cri)):
		cri[i] -= 1
		if cri[i] == 0:
			cri_dead += 1

	return cri_dead > 0 and par_dead > 0

def meet(has_meeting):
	if has_meeting:
		offset = 0
		for i in range(len(cri)):
			if cri[i-offset] == 0 and randint(0, 100) < z:
				cri.pop(i-offset)
				cri_life.pop(i-offset)
				offset += 1

def increase():
	for i in range(len(cri)):
		if cri[i] == 0 and randint(0, 100) < val_c:
			cri.append(cvi_c)
			cri_life.append(cvi_c)
	for i in range(len(par)):
		if par[i] == 0 and randint(0, 100) < val_p:
			par.append(cvi_p)
			par_life.append(cvi_p)
			par_last.append(0)

def mutate(has_meeting):
	offset = 0
	if has_meeting:
		for i in range(len(cri)):
			j = i-offset
			if cri[j] == 0:
				r = randint(0, 100)
				if r < xc:
					cri_life[j] += 1
				elif r > xc+yc:
					if cri_life[j] > 1:
						cri_life[j] -= 1
					else:
						cri.pop(j)
						cri_life.pop(j)
						offset += 1

	else:
		for i in range(len(par)):
			j = i-offset
			if par[j] == 0:
				r = randint(0, 100)
				if r < xc:
					par_life[j] += 1
				elif r > xc+yc:
					if par_life[j] > 1:
						par_life[j] -= 1
					else:
						par.pop(j)
						par_life.pop(j)
						par_last.pop(j)
						offset += 1


def give_birth():
	for i in range(len(cri)):
		if cri[i] == 0:
			cri[i] = cri_life[i]
	offset = 0
	for i in range(len(par)):
		if par[i-offset] == 0:
			par_last[i-offset] += 1
			if par_last[i-offset] == nc:
				par.pop(i-offset)
				par_life.pop(i-offset)
				par_last.pop(i-offset)
				offset += 1
			else:
				par[i-offset] = par_life[i-offset]

def game_over():
	return len(par_life) == 0 or len(cri_life) == 0

def display(round):
	print "\n\nIter %s" % round
	print '   parasites:'
	s = 'Years left   '
	for i in range(len(par)):
		s += ' ' + str(par[i])
	print s
	s = 'Lifes   '
	for i in range(len(par)):
		s += ' ' + str(par_life[i])
	print s
	print '   crickets:'
	s = 'Years left   '
	for i in range(len(cri)):
		s += ' ' + str(cri[i])
	print s
	s = 'Lifes   '
	for i in range(len(cri)):
		s += ' ' + str(cri_life[i])
	print s

def live():
	for i in range(max_steps):
		has_meeting = pass_year()
		meet(has_meeting)
		increase()
		mutate(has_meeting)
		give_birth()
		display(i)
		if game_over():
			break

live()