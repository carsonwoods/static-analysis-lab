/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__char_realloc_83_bad.cpp
Label Definition File: CWE401_Memory_Leak.c.label.xml
Template File: sources-sinks-83_bad.tmpl.cpp
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource: realloc Allocate data using realloc()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 83 Data flow: data passed to class constructor and destructor by declaring the class object on the stack
 *
 * */
#ifndef OMITBAD

#include "std_testcase.h"
#include "CWE401_Memory_Leak__char_realloc_83.h"

namespace CWE401_Memory_Leak__char_realloc_83
{
CWE401_Memory_Leak__char_realloc_83_bad::CWE401_Memory_Leak__char_realloc_83_bad(char * dataCopy)
{
    data = dataCopy;
    /* POTENTIAL FLAW: Allocate memory on the heap */
    data = (char *)realloc(data, 100*sizeof(char));
    /* Initialize and make use of data */
    strcpy(data, "A String");
    printLine(data);
}

CWE401_Memory_Leak__char_realloc_83_bad::~CWE401_Memory_Leak__char_realloc_83_bad()
{
    /* POTENTIAL FLAW: No deallocation */
    ; /* empty statement needed for some flow variants */
}
}
#endif /* OMITBAD */
