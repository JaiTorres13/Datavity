import ply.lex as lex

import ply.yacc as yacc
from src import Datavity as fe


# Command guide to help user understand this code
def help():
    print('''Datavity: The Feature Engineering language.\n
            Display the title of each columns: features() \n 
            Substitute missing data with max, min, mean or zero: clean(MAX || MIN || MEAN || ZERO) \n
            Display 3 histograms, without change, applying linear scaling and log scaling: transform(feature_name) \n
            Calculate RMSE & display a scatter plot of first feature vs second feature: rmse(MAX || MIN || MEAN || ZERO , feature_name , feature_name) \n
            Exit Datavity: exit
    ''')


# lexer part
reserved = {
    '(': 'LPAREN',
    ')': 'RPAREN',
    'features': 'FEATURES',
    'clean': 'CLEAN',
    'transform': 'TRANSFORM',
    'rmse': 'RMSE'
}
tokens = [
             'ID',
             'COMMA',

         ] + (list(reserved.values()))
t_ignore = ' '


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t


t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r'\,'


def t_error(t):
    print("Illegal character '%s" % t.value[0])
    t.lexer.skip(1)


lexer = lex.lex()


# Parser part
# The following methods follow from top to bottom

# This is the first method to work for each input from user
def p_define(p):
    '''define : features
            | clean
            | transform
            | rmse'''

    # p works as a list and each index is a token from lexer
    p[0] = p[1]


# reachable when the user enters: features()
def p_features(p):
    'features : FEATURES LPAREN RPAREN'
    fe.FEATURES()


# reachable when the user enters for example: clean(ZERO)
def p_clean(p):
    'clean : CLEAN LPAREN ID RPAREN'
    # treat p as if it were a list; each separate word is a cell in the list.
    if p[3] == 'MAX' or p[3] == 'MIN' or p[3] == 'MEAN' or p[3] == 'ZERO':
        fe.CLEAN(p[3])
    else:
        print("Error: parameter expected to be MAX, MIN, MEAN or ZERO. Instead of '" + p[3] + "'")


# reachable when the user enters for example: transform(price)
def p_transform(p):
    'transform : TRANSFORM LPAREN ID RPAREN'
    # It makes sure that ID is from the features shown in the fe.FEATURES() method
    if not fe.TRANSFORM(p[3]):
        print("Error: '" + p[3] + "' is not part of the numeric feature")


# reachable when the user enters for example: rmse(ZERO, price, losses)
def p_rmse(p):
    'rmse : RMSE LPAREN ID COMMA ID COMMA ID RPAREN'
    if (p[3] == 'MAX' or p[3] == 'MIN' or p[3] == 'MEAN' or p[3] == 'ZERO') and p[5] != p[7]:
        if not fe.RMSE(p[3], p[5], p[7]):
            print("Error: the feature(s) are not part of the numeric features")
    else:
        print("Error: invalid parameter(s)")


# if none of the previous methods from parser were reached, then this one activates by default
def p_error(p):
    print('Syntax error in code provided!', s)


parser = yacc.yacc()

# Runs until user enters 'exit' to end the code
while True:
    try:
        s = input('Datavity > ')
    except EOFError:
        break
    if s == "exit":
        print('Thank you for using Datavity!')
        break
    if s == 'help':
        help()
        continue
    if not s: continue
    result = parser.parse(s)
