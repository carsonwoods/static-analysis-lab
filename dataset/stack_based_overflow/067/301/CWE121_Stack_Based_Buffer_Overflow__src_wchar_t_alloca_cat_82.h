/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82.h
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__src.label.xml
Template File: sources-sink-82.tmpl.h
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Initialize data as a large string
 * GoodSource: Initialize data as a small string
 *    BadSink : Copy data to string using wcscat
 * Flow Variant: 82 Data flow: data passed in a parameter to an virtual method called via a pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82
{

class CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82_base
{
public:
    /* pure virtual function */
    virtual void action(wchar_t * data) = 0;
};

#ifndef OMITBAD

class CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82_bad : public CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82_base
{
public:
    void action(wchar_t * data);
};

#endif /* OMITBAD */

#ifndef OMITGOOD

class CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82_goodG2B : public CWE121_Stack_Based_Buffer_Overflow__src_wchar_t_alloca_cat_82_base
{
public:
    void action(wchar_t * data);
};

#endif /* OMITGOOD */

}
