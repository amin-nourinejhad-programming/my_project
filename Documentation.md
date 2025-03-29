## Welcome To My Documentation 😊
---------------------------------------
### Task 1
----------------------------------

#### Concise Result Explanation

|Test Name    |    Status  |  
|-------------|------------|    
| **Test 1**  | ✅ passed |
| **Test 2**  | ❌ Failed |
---------------------------------------------------------------------
####  why Test 1 has passed status?

Because it found "Test2" in one of the lines. And it is replaced with Testdata2
and added to "new_one.txt" .

There are no errors.

---------------------------------------------------------------------------
####  why Test 2 has failed status?

Test 2 expects that the file new_one.txt to be empty.
So it Uses the built-in Robot Framework keyword **File Should Be Empty**  to
check it.

Since the file is not empty the test leads to failure.

------------------------------------------------------------------------
####  How the keyword 'My New Keyword' works?
1- it reads the 'data.text'.

2- it splits the content of the data into lines.

3- it loops through each line to find 'Test2'.

4- if it does not find the 'Test2', it skips the line and continue. and if it
   finds the 'Test2', it logs a Warning message.

5- it replaces 'test2' with 'Testdata2'.

6- it stored the modified line into a Variable.

7- it writes the updated line in 'new_one.text'.

-----------------------------------------------------------------------------
#### What is Much meaningful name for 'My New Keyword' ?
In my opinion it can be:  **"find_string_then_replace"**.

-----------------------------------------------------------------------------
## Thank you for your time🙏
