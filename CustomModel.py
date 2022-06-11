def mapping_fun(df, *args):
    size = len(df.columns)
    output = args[size]
    for i in range(0, size):
        output += df[df.columns[i]] * args[i]
    return output


class CustomModel:
    def __init__(self, parameters):
        self.predicate = mapping_fun
        self.parameters = parameters

    def run(self, X):
        return self.predicate(X, *self.parameters)






