/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__int_calloc_84_goodB2G.cpp
Label Definition File: CWE401_Memory_Leak.c.label.xml
Template File: sources-sinks-84_goodB2G.tmpl.cpp
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource: calloc Allocate data using calloc()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 84 Data flow: data passed to class constructor and destructor by declaring the class object on the heap and deleting it after use
 *
 * */
#ifndef OMITGOOD

#include "std_testcase.h"
#include "CWE401_Memory_Leak__int_calloc_84.h"

namespace CWE401_Memory_Leak__int_calloc_84
{
CWE401_Memory_Leak__int_calloc_84_goodB2G::CWE401_Memory_Leak__int_calloc_84_goodB2G(int * dataCopy)
{
    data = dataCopy;
    /* POTENTIAL FLAW: Allocate memory on the heap */
    data = (int *)calloc(100, sizeof(int));
    /* Initialize and make use of data */
    data[0] = 5;
    printIntLine(data[0]);
}

CWE401_Memory_Leak__int_calloc_84_goodB2G::~CWE401_Memory_Leak__int_calloc_84_goodB2G()
{
    /* FIX: Deallocate memory */
    free(data);
}
}
#endif /* OMITGOOD */
