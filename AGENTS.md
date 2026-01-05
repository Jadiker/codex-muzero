# AGENTS.md
This file contains instructions for autonomous agents for understanding and editing the code.

Any code you add should pass Pylance (at default "standard" level). Use `cast` as needed rather than `# ignore` to deal with Pylance warnings.


For each task that you complete or question that you answer, write a `YYYY-MM-DD-(HH:MM)-name-of-report.md` report with a descriptive name inside `agent_explain/` documenting
  1. The current date
  2. The current time in UTC
  3. Your (agent) name and what model you are powered by
  4. A summary of what you did
  5. The exact words of what the user asked you to do.
  6. A list of everything you did while attempting to accomplish what the user asked. For each item in the list, explain why you did it, what other options (if any) you considered or attempted that didn't work, and any other information you think might be helpful for fully explaining your work and reasoning
  7. Useful data and any other information that may be helpful for agents or humans who may attempt to build upon your work later
  8. Any questions you may have for the developer.
   
ANY PROPOSED CHANGES OR ANSWERS YOU MAKE WITHOUT A PROPERLY NAMED AND FILLED OUT REPORT WILL BE REJECTED. ALWAYS WRITE A REPORT.

Only install and use packages in the `.venv` environment.