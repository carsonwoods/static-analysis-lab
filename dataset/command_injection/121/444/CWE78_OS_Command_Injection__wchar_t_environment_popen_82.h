/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__wchar_t_environment_popen_82.h
Label Definition File: CWE78_OS_Command_Injection.one_string.label.xml
Template File: sources-sink-82.tmpl.h
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: environment Read input from an environment variable
 * GoodSource: Fixed string
 *    BadSink : Execute command in data using popen()
 * Flow Variant: 82 Data flow: data passed in a parameter to an virtual method called via a pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
#define FULL_COMMAND L"%WINDIR%\\system32\\cmd.exe /c dir "
#else
#include <unistd.h>
#define FULL_COMMAND L"/bin/sh ls -la "
#endif

namespace CWE78_OS_Command_Injection__wchar_t_environment_popen_82
{

class CWE78_OS_Command_Injection__wchar_t_environment_popen_82_base
{
public:
    /* pure virtual function */
    virtual void action(wchar_t * data) = 0;
};

#ifndef OMITBAD

class CWE78_OS_Command_Injection__wchar_t_environment_popen_82_bad : public CWE78_OS_Command_Injection__wchar_t_environment_popen_82_base
{
public:
    void action(wchar_t * data);
};

#endif /* OMITBAD */

#ifndef OMITGOOD

class CWE78_OS_Command_Injection__wchar_t_environment_popen_82_goodG2B : public CWE78_OS_Command_Injection__wchar_t_environment_popen_82_base
{
public:
    void action(wchar_t * data);
};

#endif /* OMITGOOD */

}
