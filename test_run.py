from BayesNet import BayesNet


obj = BayesNet()
obj.load_from_bifxml("testing/lecture_example.BIFXML")
for var in obj.get_all_variables():
    print(obj.get_cpt(var))
    print('*** NEXT *** Above', var)


