# SQL Review
## SQL Concepts
### Data Definition Language (DDL)
Define the different structures in a database. DDL statements create, modify, and remove database objects such as tables

### Data Manipulation Language (DML) 
Used for adding, deleting, and modifying data in a database.
    * SELECT column_name(s)
FROM table_name
JOIN table2 on keys
WHERE condition
GROUP BY column_name(s)
HAVING condition
ORDER BY column_name(s)
LIMITS

### Primary Key
a field or collection of fields to uniquely identify all table records. It must contain a unique value for each row of data, and cannot contain null values.

### Unique Key
A UNIQUE constraint ensures that all values in a column are different. This provides uniqueness for the column(s) and helps identify each row uniquely. Unlike primary key, there can be multiple unique constraints defined per table.

### Foreign Key
A field or collection of fields in a table that essentially refer to the PRIMARY KEY in another table.

### Relationships
Relations or links between entities that have something to do with each other.
    * Inner Join
Selects records that have matching values in both tables.
    * Left Join
Returns all records from the left table, and the matched records from the right table. The result is NULL from the right side, if there is no match.

### Index
A data structure that provides quick lookup of data in a column or columns of a table. It enhances the speed of operations accessing data from a database table at the cost of additional writes and memory to maintain the index data structure.

### Subquery
A query within another query, also known as nested query or inner query . It is used to restrict or enhance the data to be queried by the main query.

### Common Table Expression (CTE)
A CTE allows you to define a temporary named result set that available temporarily in the execution scope of a statement such as SELECT, INSERT, UPDATE, DELETE, or MERGE. We prefer to use common table expressions rather than to use subqueries because common table expressions are more readable. We also use CTE in the queries that contain analytic functions (or window functions)

### View
A virtual table which consists of a subset of data contained in a table. Views does not contain real data, so it takes less space to store. And, it is easier to share data to multiple users.

### String Formatting
* LEFT, RIGHT
* UPPER, LOWER
* REVERSE: reverses a string
* TRIM: removes leading and trailing spaces from a string
* POSITION: returns the index of the first occurrence of a substring in a string
* SUBSTR: extracts a substring from a string by index and length
* CONCAT: adds two or more expressions together. returns null if null values contain
* CONCAT_WS: adds two or more expressions together with a separator. ignore null values
* COALESCE/ISNULL: returns the first non-null value in a list, or impute missing values
* Datetime Manipulation
* DATE: format YYYY-MM-DD
* DATETIME: format YYYY-MM-DD HH:MI:SS
* YEAR, MONTH, DAY
* DATEDIFF: Returns the number of date or time btw two dates
* DATEADD: returns a new datetime value by adding an interval

## Practical Questions
### Question: find/remove duplicates
1. Distinct: remove duplicates
    * copy all values to a temporary table
    * sort duplicates and return unique items
    * optimize memory usage due to temp table
3. Groupby: remove duplicates
    * scan the full table, storing values in a hashtable
    * return sorted keys of the hashtable
    * requires a large amount of memory
    * slower than `Distinct` due to extra storing for returns
    * slower in DFS due to partition keys from multiple clusters
5. Union: remove duplicates when concat tables
6. Aggregation counts: find/remove duplicates

| Id | Email   |
| -- | ------- |
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
Answer:
```sql=
-- find duplicates
select Email
from Person
group by Email
having count(Email) > 1

-- remove duplicates
select * from
(
    select Email, rank() over (partition by Email order by Email) as n
    from Person
) x
where x.n = 1
```

### Questions: ranking
The ranking functions in SQL return the row number of each record starting from 1.
* `ROW_NUMBER`: return randomly row numbers for same values present in two different rows.
* `RANK`: return the same row numbers for same values, but will skip positions for next
* `DENSE_RANK`: return the same row numbers for same values, and not skip positions for next

### Question: self-join
Write a SQL query that finds out employees who earn more than their managers. For the above table, Joe is the only employee who earns more than his manager.

| Id | Name  | Salary | ManagerId |
| ---|-------- | -------- | -------- |
| 1  | Joe   | 70000  | 3         |
| 2  | Henry | 80000  | 4         |
| 3  | Sam   | 60000  | NULL      |
| 4  | Max   | 90000  | NULL      |

Answer:
```sql=
select a.Name as Employee
from Employee as a
join Employee as b on a.ManagerID = b.Id
where a.Salary > b.Salary
```

### Question: find medican
Write a query to find medican GPA for each grade
```sql=
-- `percentile_cont` is not aggre function
-- `partition by` is not necessary
select ID, grade
, percentile_cont(0.5) within group (order by GPA) over (partition by grade) as med_gpa 
from A
```

### Question: Longest not Run
Write a query to compute the longest days not run for each user

Input:
| ID  | Day | Run |
|-----|-----|-----|
| 1   |  1 |   0 | 
| 1   |  2 |   0 | 
| 1   |  3 |   1 | 
| 1   |  4 |   0 |
| 1   |  5 |   0 |
| 1   |  6 |   0 |
| 1   |  7 |   1 |
| 2   |  1 |   1 | 
| 2   |  2 |   0 | 
| 2   |  3 |   0 |

Output:
| ID  | Day |
|-----|-----|
| 1   |  3 |
| 2   |  2 |

Answer:
```sql=
select ID, max(cumsum)-1
from
(
    select ID, Day, sum(Run) over (partition by ID order by Day) as cumsum
    from A
) x
group by ID
```

### Question
Write a query to output the start and dates of projects listed by the number of days took to complete in ascending order. If the End_Date of the tasks are consecutive, then they are part of the same project.

Input:
| ID  | start_date | end_date |
|----------|--------|--------|
| 1   |    2015-10-01 |   2015-10-02 | 
| 2   |    2015-10-02 |   2015-10-03 | 
| 3   |    2015-10-03 |   2015-10-04 | 
| 4   |    2015-10-13 |   2015-10-14 | 
| 5   |    2015-10-14 |   2015-10-15 | 
| 6   |    2015-10-28 |   2015-10-29 | 
| 7   |    2015-10-30 |   2015-10-31 |

Output:
| start_date | end_date |
|--------|--------|
|    2015-10-28 |   2015-10-29 | 
|    2015-10-30 |   2015-10-31 | 
|    2015-10-13 |   2015-10-15 |
|    2015-10-01 |   2015-10-04 |

```sql=
with a as (
select row_number() over (order by start_date) as id, start_date
from x
where start_date not in (select end_date from x)
),

b as (
select row_number() over (order by end_date) as id, end_date
from x
where end_date not in (select start_date from x)
),

select a.start_date, b.end_date
from a
join b on b.id = a.id
order by diffdate(day, a.start_date, b.start_date)
```

### Question: Two Tables Comparsion
```sql=
--- NOT IN
SELECT Cat_ID
FROM Category_A  WHERE Cat_ID NOT IN (SELECT Cat_ID FROM Category_B)

-- NOT EXISTS
SELECT A.Cat_ID
FROM Category_A A WHERE NOT EXISTS (SELECT B.Cat_ID FROM Category_B B WHERE B.Cat_ID = A.Cat_ID)

-- LEFT JOIN
SELECT A.Cat_ID
FROM Category_A A 
LEFT JOIN Category_B B ON A.Cat_ID = B.Cat_ID
WHERE B.Cat_ID IS NULL

-- EXCEPT
SELECT A.Cat_ID
FROM Category_A A 
EXCEPT 
SELECT B.Cat_ID
FROM Category_B B
```

We conclude, first, that using the SQL `NOT EXISTS` or the `LEFT JOIN` commands are the best choice from all performance aspects. We tried also to add an index on the joining column on both tables, where the query that uses the `EXCEPT` command enhanced clearly and showed better performance. In addition, `NOT IN` will return nothing when table B has NULL value.