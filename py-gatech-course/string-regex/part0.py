
# coding: utf-8

# # Part 0 of 2: Simple string processing review

# In[1]:


text = "sgtEEEr2020.0"


# In[2]:


# Strings have methods for checking "global" string properties
print("1.", text.isalpha())

# These can also be applied per character
print("2.", [c.isalpha() for c in text])


# In[3]:


# Here are a bunch of additional useful methods
print("BELOW: (global) -> (per character)")
print(text.isdigit(), "-->", [c.isdigit() for c in text])
print(text.isspace(), "-->", [c.isspace() for c in text])
print(text.islower(), "-->", [c.islower() for c in text])
print(text.isupper(), "-->", [c.isupper() for c in text])
print(text.isnumeric(), "-->", [c.isnumeric() for c in text])


# **Exercise 0** (1 point). Create a new function that checks whether a given input string is a properly formatted social security number, i.e., has the pattern, `XXX-XX-XXXX`, _including_ the separator dashes, where each `X` is a digit. It should return `True` if so or `False` otherwise.

# In[ ]:

import re
def is_ssn(s):
    regex = '\d{3}-\d{2}-\d{3}'
    pmatch = re.compile(regex)
    print(pmatch.search(s))
    return pmatch.search(s)


# In[ ]:


# Test cell: `is_snn_test`
assert is_ssn('832-38-1847')
assert not is_ssn('832 -38 -  1847')
assert not is_ssn('832-bc-3847')
assert not is_ssn('832381847')
assert not is_ssn('8323-8-1847')
assert not is_ssn('abc-de-ghij')
print("\n(Passed!)")


# # Regular expressions
# 
# Exercise 0 hints at the general problem of finding patterns in text. A handy tool for this problem is Python's Regular Expression module, `re`.
# 
# A _regular expression_ is a specially formatted pattern, written as a string. Matching patterns with regular expressions has 3 steps:
# 
# 1. You come up with a pattern to find.
# 2. You compile it into a _pattern object_.
# 3. You apply the pattern object to a string to find _matches_, i.e., instances of the pattern within the string.
# 
# As you read through the examples below, refer also to the [regular expression HOWTO document](https://docs.python.org/3/howto/regex.html) for many more examples and details.

# In[ ]:


import re


# ## Basics
# 
# Let's see how this scheme works for the simplest case, in which the pattern is an *exact substring*. In the following example, suppose want to look for the substring `'fox'` within a larger input string.

# In[ ]:


pattern = 'fox'
pattern_matcher = re.compile(pattern)

input_string = 'The quick brown fox jumps over the lazy dog'
matches = pattern_matcher.search(input_string)
print(matches)


# Observe that the returned object, `matches`, is a special object. Inspecting the printed output, notice that the matching text, `'fox'`, was found and located at positions 16-18 of the `input_string`. Had there been no matches, then `.search()` would have returned `None`, as in this example:

# In[ ]:


print(pattern_matcher.search("This input has a FOX, but it's all uppercase and so won't match."))


# You can also write code to query the `matches` object for more information.

# In[ ]:


print(matches.group())
print(matches.start())
print(matches.end())
print(matches.span())


# **Module-level searching.** For infrequently used patterns, you can also skip creating the pattern object and just call the module-level search function, `re.search()`.

# In[ ]:


matches_2 = re.search('jump', input_string)
assert matches_2 is not None
print ("Found", matches_2.group(), "@", matches_2.span())


# **Other Search Methods.** Besides `search()`, there are several other pattern-matching procedures:
# 
# 1. `match()`    - Determine if the regular expression (RE) matches at the beginning of the string.
# 2. `search()`   - Scan through a string, looking for any location where this RE matches.
# 3. `findall()`  - Find all substrings where the RE matches, and returns them as a list.
# 4. `finditer()` - Find all substrings where the RE matches, and returns them as an iterator.
# 
# We'll use several of these below; again, refer to the [HOWTO](https://docs.python.org/3/howto/regex.html) for more details.

# ## A pattern language
# 
# An exact substring is one kind of pattern, but the power of regular expressions is that it provides an entire "_mini-language_" for specifying more general patterns.
# 
# To start, read the section of the HOWTO on ["Simple Patterns"](https://docs.python.org/3/howto/regex.html#simple-patterns). We highlight a few constructs below.

# In[ ]:


# Metacharacter classes
vowels = '[aeiou]'

print(f"Scanning `{input_string}` for vowels, `{vowels}`:")
for match_vowel in re.finditer(vowels, input_string):
    print(match_vowel)


# In[ ]:


# Counts: For instance, two or more consecutive vowels:
two_or_more_vowels = vowels + '{2,}'
print(f"Pattern: {two_or_more_vowels}")
print(re.findall(two_or_more_vowels, input_string))


# In[ ]:


# Wildcards
cats = "ca+t"
print(re.search(cats, "is this a ct?"))
print(re.search(cats, "how about this cat?"))
print(re.search(cats, "and this one: caaaaat, yes or no?"))


# In[ ]:


# Special operator: "or"
adjectives = "lazy|brown"
print(f"Scanning `{input_string}` for adjectives, `{adjectives}`:")
for match_adjective in re.finditer(adjectives, input_string):
    print(match_adjective)


# In[ ]:


# Predefined character classes
three_digits = '\d\d\d'
print(re.findall(three_digits, "My number is 555-123-4567"))


# > In the previous example, notice that the pattern search proceeds from left-to-right and does not return overlaps: here, the matcher returns `456` but not `567`. In fact, this case is an instance of the default [_greedy behavior_](https://docs.python.org/3/howto/regex.html#greedy-versus-non-greedy) of the matcher.

# **The backslash plague.** In the "three-digits" example, we used the predefined metacharacter class, `'\d'`, to match slashes. But what if you want to match a _literal_ slash? The HOWTO describes how things can get out of control in its subsection on ["The Backslash Plague"](https://docs.python.org/3/howto/regex.html#the-backslash-plague), which occurs because the Python interpreter processes backslashes in string literals (e.g., so that `\t` expands to a tab character and `\n` to a newline) while the regular expression processor also gives backslashes meaning (e.g., so that `\d` is a digit metaclass).
# 
# For example, suppose you want to look for the text string, `\section`, in some input string. Which of the following will match it? Recall that `\s` is a predefined metacharacter class that matches any whitespace character.

# In[ ]:


input_with_slash_section = "This string contains `\section`, which we would like to match."

print(f"Searching: {input_with_slash_section}")

print(re.search("\section", input_with_slash_section))
print(re.search("\\section", input_with_slash_section))
print(re.search("\\\\section", input_with_slash_section))


# To help mitigate this case, Python provides a special type of string called a _raw string_, which is a string literal prefixed by the letter `r`. For such strings, the Python interpreter will not process the backslash.
# 
# > Although the interpreter won't process the backslash, the regular expression processor will do so. As such, the pattern string still needs _two_ slashes, as shown below.

# In[ ]:


print(re.search(r"\section", input_with_slash_section))
print(re.search(r"\\section", input_with_slash_section))
print(re.search(r"\\\\section", input_with_slash_section))


# Indeed, it is common style to always use raw strings for regular expression patterns, as we'll do in the examples that follow.

# **Creating pattern groups.** Another handy construct are [_pattern groups_](https://docs.python.org/3/howto/regex.html#grouping), as we show in the next code cell.
# 
# Suppose we have a string that we know contains a name of the form, "(first) (middle) (last)", where the middle name is _optional_. We can use pattern groups to isolate each component of the name and tag the middle name as optional using the "zero-or-one" metacharacter, `'?'`.
# 
# The group itself is a subpattern enclosed within parentheses. When a match is found, we can extract the groups by calling `.groups()` on the match object, which returns a tuple of all matched groups.
# 
# > To make this pattern more readable, we have also used Python's multiline string literal combined with the [`re.VERBOSE` option](https://docs.python.org/2/library/re.html#re.VERBOSE), which then allows us to include whitespace and comments as part of the pattern string.

# In[ ]:


# Make the expression more readable with a re.VERBOSE pattern
re_names2 = re.compile(r'''^              # Beginning of string
                           ([a-zA-Z]+)    # First name
                           \s+            # At least one space
                           ([a-zA-Z]+\s)? # Optional middle name
                           ([a-zA-Z]+)    # Last name
                           $              # End of string
                        ''',
                        re.VERBOSE)
print(re_names2.match('Rich Vuduc').groups())
print(re_names2.match('Rich S Vuduc').groups())
print(re_names2.match('Rich Salamander Vuduc').groups())


# **Tagging pattern groups.** You can also name pattern groups, which helps make your extraction code a bit more readable.

# In[ ]:


# Named groups
re_names3 = re.compile(r'''^
                           (?P<first>[a-zA-Z]+)
                           \s
                           (?P<middle>[a-zA-Z]+\s)?
                           \s*
                           (?P<last>[a-zA-Z]+)
                           $
                        ''',
                        re.VERBOSE)
print(re_names3.match('Rich Vuduc').group('first'))
print(re_names3.match('Rich S Vuduc').group('middle'))
print(re_names3.match('Rich Salamander Vuduc').group('last'))


# **A regular expression debugger.** Regular expressions can be tough to write and debug, but thankfully, there are several online tools to help! See, for instance, [regex101](https://regex101.com/), [pythex](https://pythex.org/), [regexr](https://regexr.com/), or [debuggex](https://www.debuggex.com/). These all allow you to supply some sample input text and test what your pattern does in real time.

# ## Email addresses
# 
# In the next exercise, you'll apply what you've read and learned about regular expressions to build a pattern matcher for email addresses. Again, if you haven't looked through the HOWTO yet, take a moment to do that!
# 
# Although there is a [formal specification of what constitutes a valid email address](https://tools.ietf.org/html/rfc5322#section-3.4.1), for this exercise, let's use the following simplified rules.
# 
# * We will restrict our attention to ASCII addresses and ignore Unicode. If you don't know what that means, don't worry about it---you shouldn't need to do anything special given our code templates, below.
# * An email address has two parts, the username and the domain name. These are separated by an `@` character.
# * A username **must begin with an alphabetic** character. It may be followed by any number of additional _alphanumeric_ characters or any of the following special characters: `.` (period), `-` (hyphen), `_` (underscore), or `+` (plus).
# * A domain name **must end with an alphabetic** character. It may consist of any of the following characters: alphanumeric characters, `.` (period), `-` (hyphen), or `_` (underscore).
# * Alphabetic characters may be uppercase or lowercase.
# * No whitespace characters are allowed.
# 
# Valid domain names usually have additional restrictions, e.g., there are a limited number of endings, such as `.com`, `.edu`, and so on. However, for this exercise you may ignore this fact.

# **Exercise 1** (2 points). Write a function `parse_email` that, given an email address `s`, returns a tuple, `(user-id, domain)` corresponding to the user name and domain name.
# 
# For instance, given `richie@cc.gatech.edu` it should return `('richie', 'cc.gatech.edu')`.
# 
# Your function should parse the email only if it exactly matches the email specification. For example, if there are leading or trailing spaces, the function should *not* match those. See the test cases for examples.
# 
# If the input is not a valid email address, the function should raise a `ValueError`.
# 
# > The requirement, "raise a `ValueError`" refers to a technique for handling errors in a program known as _exception handling_. The Python documentation covers [exceptions](https://docs.python.org/3/tutorial/errors.html) in more detail, including [raising `ValueError` objects](https://docs.python.org/3/tutorial/errors.html#raising-exceptions).

# In[ ]:


def parse_email (s):
    """Parses a string as an email address, returning an (id, domain) pair."""
    email_regex=re.compile('^([a-zA-Z._+-]*)([\@{1}])([a-zA-Z._-]*[^0-9])$')
    email_regex_tokens=email_regex.match(s)
    try:
        print(email_regex_tokens)
        #if email_regex_tokens:
        return(email_regex_tokens.groups()[0], email_regex_tokens.groups()[2])
    except:
        raise ValueError


# In[ ]:


# Test cell: `parse_email_test`

def pass_case(u, d):
    s = u + '@' + d
    msg = "Testing valid email: '{}'".format(s)
    print(msg)
    assert parse_email(s) == (u, d), msg
    
pass_case('richie', 'cc.gatech.edu')
pass_case('bertha_hugely', 'sampson.edu')
pass_case('JKRowling', 'Huge-Books.org')
pass_case('what-do-you-know+not-much', 'gmail.com')

def fail_case(s):
    msg = "Testing invalid email: '{}'".format(s)
    print(msg)
    try:
        parse_email(s)
    except ValueError:
        print("==> Correctly throws an exception!")
    else:
        raise AssertionError("Should have, but did not, throw an exception!")
        
fail_case('x @hpcgarage.org')
fail_case('   quiggy.smith38x@gmail.com')
fail_case('richie@cc.gatech.edu  ')
fail_case('4test@gmail.com')
fail_case('richie@cc.gatech.edu7')


# ## Phone numbers

# **Exercise 2** (2 points). Write a function to parse US phone numbers written in the canonical "(404) 555-1212" format, i.e., a three-digit area code enclosed in parentheses followed by a seven-digit local number in three-hyphen-four digit format. It should also **ignore** all leading and trailing spaces, as well as any spaces that appear between the area code and local numbers. However, it should **not** accept any spaces in the area code (e.g., in '(404)') nor should it in the seven-digit local number.
# 
# For example, these would be considered valid phone number strings:
# ```python
#     '(404) 121-2121'
#     '(404)121-2121     '
#     '   (404)      121-2121'
# ```
# 
# By contrast, these should be rejected:
# ```python
#     '404-121-2121'
#     '(404)555 -1212'
#     ' ( 404)121-2121'
#     '(abc) 555-12i2'
# ```
# 
# It should return a triple of strings, `(area_code, first_three, last_four)`. 
# 
# If the input is not a valid phone number, it should raise a `ValueError`.

# In[ ]:


def parse_phone1 (s):
    phone_regex=re.compile('\s*\((?P<areacode>\d{3})\)\s*(?P<first_three>\d{3})\-(?P<last_four>\d{4})\s*')
    phone_regex_tokens=phone_regex.match(s)
    #print('Current String: {0}\nRegex Tokens:{1}'.format(s, phone_regex_tokens))
    try:
        return (phone_regex_tokens.group('areacode'),
        phone_regex_tokens.group('first_three'),
        phone_regex_tokens.group('last_four'))
    except:
        raise ValueError


# In[ ]:


# Test cell: `parse_phone1_test`

def rand_spaces(m=5):
    from random import randint
    return ' ' * randint(0, m)

def asm_phone(a, l, r):
    return rand_spaces() + '(' + a + ')' + rand_spaces() + l + '-' + r + rand_spaces()

def gen_digits(k):
    from random import choice # 3.5 compatible; 3.6 has `choices()`
    DIGITS = '0123456789'
    return ''.join([choice(DIGITS) for _ in range(k)])

def pass_phone(p=None, a=None, l=None, r=None):
    if p is None:
        a = gen_digits(3)
        l = gen_digits(3)
        r = gen_digits(4)
        p = asm_phone(a, l, r)
    else:
        assert a is not None and l is not None and r is not None, "Need to supply sample solution."
    msg = "Should pass: '{}'".format(p)
    print(msg)
    p_you = parse_phone1(p)
    assert p_you == (a, l, r), "Got {} instead of ('{}', '{}', '{}')".format(p_you, a, l, r)
    
def fail_phone(s):
    msg = "Should fail: '{}'".format(s)
    print(msg)
    try:
        p_you = parse_phone1(s)
    except ValueError:
        print("==> Correctly throws an exception.")
    else:
        raise AssertionError("Failed to throw a `ValueError` exception!")


# Cases that should definitely pass:
pass_phone('(404) 121-2121', '404', '121', '2121')
pass_phone('(404)121-2121', '404', '121', '2121')
pass_phone('   (404) 121-2121', '404', '121', '2121')
pass_phone(' (404)121-2121    ', '404', '121', '2121')
for _ in range(5):
    pass_phone()
    
fail_phone("404-121-2121")
fail_phone('(404)555 -1212')
fail_phone(" ( 404)121-2121")
fail_phone("(abc) 555-1212")
fail_phone("(678) 555-12i2")


# **Exercise 3** (3 points). Implement an enhanced phone number parser that can handle any of these patterns.
# 
# * (404) 555-1212
# * (404) 5551212
# * 404-555-1212
# * 404-5551212
# * 404555-1212
# * 4045551212
# 
# As before, it should not be sensitive to leading or trailing spaces. Also, for the patterns in which the area code is enclosed in parentheses, it should not be sensitive to the number of spaces separating the area code from the remainder of the number.

# In[ ]:


def parse_phone2 (s):
    phone_regex_2=re.compile('\s*(?P<areacode>\d{3})\S*\-*(?P<first_three>\d{3})\-*(?P<last_four>\d{4})\s*')
    s1=s.lstrip()
    #print(s1[0], s1[3])
    if s1[0] == '(' and s1[4]==')':
        phone_regex_2=re.compile('\s*\(*(?P<areacode>\d{3})\)*\s*\-*(?P<first_three>\d{3})\-*(?P<last_four>\d{4})\s*')

    phone_regex_tokens_2=phone_regex_2.match(s)
    print('Current String: {0}\nRegex Tokens:{1}'.format(s, phone_regex_tokens_2))
    try:
        return (phone_regex_tokens_2.group('areacode'),
        phone_regex_tokens_2.group('first_three'),
        phone_regex_tokens_2.group('last_four'))
    except:
        raise ValueError

# In[ ]:


# Test cell: `parse_phone2_test`

def asm_phone2(a, l, r):
    from random import random
    x = random()
    if x < 0.33:
        a2 = '(' + a + ')' + rand_spaces()
    elif x < 0.67:
        a2 = a + '-'
    else:
        a2 = a
    y = random()
    if y < 0.5:
        l2 = l + '-'
    else:
        l2 = l
    return rand_spaces() + a2 + l2 + r + rand_spaces()

def pass_phone2(p=None, a=None, l=None, r=None):
    if p is None:
        a = gen_digits(3)
        l = gen_digits(3)
        r = gen_digits(4)
        p = asm_phone2(a, l, r)
    else:
        assert a is not None and l is not None and r is not None, "Need to supply sample solution."
    msg = "Should pass: '{}'".format(p)
    print(msg)
    p_you = parse_phone2(p)
    assert p_you == (a, l, r), "Got {} instead of ('{}', '{}', '{}')".format(p_you, a, l, r)
    
pass_phone2("  (404)   555-1212  ", '404', '555', '1212')
pass_phone2("(404)555-1212  ", '404', '555', '1212')
pass_phone2("  404-555-1212 ", '404', '555', '1212')
pass_phone2("  404-5551212 ", '404', '555', '1212')
pass_phone2(" 4045551212", '404', '555', '1212')
    
for _ in range(5):
    pass_phone2()
    
    
def fail_phone2(s):
    msg = "Should fail: '{}'".format(s)
    print(msg)
    try:
        parse_phone2 (s)
    except ValueError:
        print ("==> Function correctly raised an exception.")
    else:
        raise AssertionError ("Function did *not* raise an exception as expected!")
        
failure_cases = ['+1 (404) 555-3355',
                 '404.555.3355',
                 '404 555-3355',
                 '404 555 3355',
                 '(404-555-1212'
                ]
for s in failure_cases:
    fail_phone2(s)
    
print("\n(Passed!)")


# **Fin!** This cell marks the end of Part 0. Don't forget to save, restart and rerun all cells, and submit it. When you are done, proceed to Parts 1 and 2.

# %%
