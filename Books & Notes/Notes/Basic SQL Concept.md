# Basic SQL Concept
### What is SQL
**Data Definition Language** (DDL) is that define the different structures in a database. DDL statements create, modify, and remove database objects such as tables

**Data Manipulation Language** (DML) is used for adding, deleting, and modifying data in a database.

SELECT column_name(s)
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

### Inner Join
Selects records that have matching values in both tables.

### Left Join
Returns all records from the left table, and the matched records from the right table. The result is NULL from the right side, if there is no match.

### Index
A data structure that provides quick lookup of data in a column or columns of a table. It enhances the speed of operations accessing data from a database table at the cost of additional writes and memory to maintain the index data structure.

### Subquery
A query within another query, also known as nested query or inner query . It is used to restrict or enhance the data to be queried by the main query.

### Entities
An entity can be a real-world object.

### Relationships
Relations or links between entities that have something to do with each other.

### View
A virtual table which consists of a subset of data contained in a table. Views does not contain real data, so it takes less space to store. And, it is easier to share data to multiple users.