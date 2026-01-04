https://github.com/kornia/kornia/blob/main/AI_POLICY.md
https://github.com/zulip/zulip/blob/main/CONTRIBUTING.md?plain=1#ai-use-policy-and-guidelines


# AI usage policy

Our goal in the Zulip project is to develop an excellent software
system. This requires careful attention to detail in every change we
integrate. Maintainer time and attention is very limited, so it's
important that changes you ask us to review represent
your _best_ work.

You can use any tools that help you understand the Zulip codebase and
write good code, including AI tools. However, as noted above, you
always need to understand and explain the changes you're proposing to
make, whether you have used an LLM as part of your process to
produce them or not. The answer to “Why is X an improvement?” should never be
“I'm not sure. The AI did it.”

**Do not submit an AI-generated PR you haven't personally understood and
tested**, as this wastes maintainers' time. PRs that appear to violate this
guideline will be closed without review.

## Using AI as a coding assistant

1. Don't skip **becoming familiar with the part of the codebase**
   you're working on. Code assistants can be a useful search engine/discovery tool in
   this process, but don't trust claims they make about how the project works.
   LLMs are often wrong, even about details that are clearly answered in our
   documentation.
2. Don't simply ask an LLM to add **code comments**, as it will likely
   produce a bunch of text that unnecessarily explains what's already
   clear from the code. If using an LLM to generate comments, be
   really specific in your request, demand succinctness, and carefully
   edit the result.

### Using AI for communication

As noted above, Zulip's contributors are expected to communicate with
intention, to avoid wasting maintainer time with long, sloppy
writing. We strongly prefer clear and concise communication about
points that actually require discussion over long AI-generated
comments.

When you use an LLM to write a message for you, it remains **your
responsibility** to read through the whole thing and make sure that it
makes sense to you and represents your ideas concisely. A good rule
of thumb is that if you can't make yourself carefully read some LLM
output that you generated, nobody else wants to read it either.

Here are some concrete guidelines for using LLMs as part of your communication
workflows.

1. When writing a pull request description, **do not include anything that's
   obvious** from looking at your changes directly (e.g., files changed, functions
   updated, etc.). Instead, focus on the _why_ behind your changes. Don't ask an
   LLM to generate a PR description on your behalf based on your code changes,
   as it will simply regurgitate the information that's already there.
1. Similarly, when responding to a pull request comment, **explain _your_
   reasoning**. Don't prompt an LLM to re-describe what can already be seen from
   the code.
1. Verify that **everything you write is accurate**, whether or not an LLM
   generated any part of it. Zulip's maintainers will be unable to review your
   contributions if you misrepresent your work (e.g., misdescribing your code
   changes, their effect, or your testing process).
1. Complete all parts of the **PR description template**, including screenshots
   and the self-review checklist. Don't simply overwrite the template with LLM
   output.
1. **Clarity and succinctness** are much more important than perfect grammar, so
   you shouldn't feel obliged to pass your writing through an LLM. If you do ask
   an LLM to clean up your writing style, be sure it does _not_ make it longer
   in the process. Demand succinctness in your prompt.
1. Quoting an LLM answer is usually less helpful than linking to
   **relevant primary sources**, like source code, reference
   documentation, or web standards. If you do need to quote an LLM
   answer in a Zulip conversation, put the answer in a [Zulip quote
   block](https://zulip.com/help/format-a-quote), to distinguish LLM
   output from your own thoughts.
