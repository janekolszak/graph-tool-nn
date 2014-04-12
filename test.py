from gtnn.generators.multilayer_perceptron import multilayer_perceptron
print("1")
n = multilayer_perceptron([4,20,2])

print("2")
for i in range(1000):
    n.forward([1,2,3,4])
print("3")
