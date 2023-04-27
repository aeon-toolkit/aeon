# Governance and decision-making

The purpose of this document is to formalize the governance process used by the `aeon`
project, to clarify how decisions are made and how the various elements of our
community interact.

`aeon` is a community-owned and community-run project. Anyone with an interest in the
project can join the community, contribute to the project design and participate in the
decision-making process.

This document establishes the decision-making structure used by `aeon` which strives to
find consensus from the community, while avoiding any deadlocks.

## Roles

### Contributors

Contributors are community members who have contributed in concrete ways to the project.
Anyone can become a contributor, and contributions can take many forms – not only code –
as detailed in the contributing guide.

### Core Developers

Core developers are community members that have made non-trivial contributions and are
trusted in continuing the development of the project and engaging with its community.
Core Developers are granted write access to the `aeon` repository and voting rights
on most project decisions. Core Developers are expected to review code contributions
and engage with topics or code they are knowledgeable about.

New Core Developers are nominated by existing Core Developers, and are subject to a
two-third majority vote of existing Core Developers. Core Developers are expected to
maintain a reasonable amount of engagement with the project, and may lose their status
after a long period of inactivity and lack of engagement with fellow developers.
Developing code, interacting with contributions and engaging with the broader community
are all valid contributions for Core Developers.

### Community Council

Community Council (CC) members are Core Developers who have additional responsibilities
to ensure the smooth running of the project. This includes strategic planning, project
infrastructure management and release management. In the event Core Developers fail to
reach consensus, the CC is the entity to resolve the issue.

CC membership is subject to a two-third majority vote of Core Developers. CC terms last
for two years, after which a new vote must be held.

### Code of Conduct Committee

Code of Conduct Committee (CoCC) members are contributors which are tasked with
maintaining the `aeon` Code of Conduct (CoC) and managing reports of breaking the CoC.
CoC members are expected to review reports of CoC violations, contact and discuss with
involved individuals and make recommendations on actions to take.

Any CoCC members involved in a CoC report or CoCC members which have a conflict of
interest regarding the report are expected to recuse themselves from the process.
If no CoCC members are available to review a report in a timely manner, the
responsibility will fall to the Community Council.

CoCC membership is subject to a two-third majority vote of Core Developers. CoCC terms
last for two years, after which a new vote must be held. CC members cannot
simultaneously be CoCC members.

## Decision-Making Process

Decisions about the future of the project are made publicly to allow discussion with
all members of the community. All non-sensitive project management discussion takes
place on the contributors channel on the project Slack and/or the issue tracker.
Occasionally, sensitive discussion and votes such as appointments will occur in private
Slack channels.

For most decisions a consensus seeking process of all interested contributors is used.
Contributors try to find a resolution that has no open objections among Core Developers.
If a reasonable amount of time has passed since the last change to a proposed
contribution, the proposal has at least one approval (+1) and no rejections (-1) from
Core Developers, it can be approved by lazy consensus. If a change is rejected, it is
expected that some form of explanation and description of conditions (if any) to
withdraw the rejection is provided.

At any point during the discussion, any Core Developer in favour of a change can call
for a vote, which will conclude two weeks from the call for the vote. Any vote must be
backed by an AEP (see following section). If no option can gather two-thirds of the
votes cast, the decision is escalated to the Community Council, which in turn will use
consensus seeking with the fallback option of a simple majority vote if no consensus
can be found.

All changes to the `aeon` code or documentation should be done via Pull Request.
By default, push rights to the `main` GitHub branch are restricted for all developers.
In emergencies, the Community Council may temporarily revoke the branch protection for
council members and make direct commits, but this should happen only in emergencies
where harm will come to the project unless timely action is taken.

### Enhancement Proposals

For all non-appointment decision-making votes, a proposal must have been made public
and discussed before the vote. Such proposal must be a consolidated document, in the
form of a “aeon Enhancement Proposal” (AEP). Having a rejection on a pull request does
not necessitate the creation of an AEP and further discussion to find consensus can be
held, but one must be created prior to any vote.

A vote does not have to be called for an AEP to be submitted. If a contributor believes
a change will be controversial, an AEP can be submitted to the community for discussion
and comment.

## Acknowledgements

Substantial portions of this document were adapted from the following projects
governance documents:

- [Scikit-learn](https://scikit-learn.org/stable/governance.html)
- [SciPy](https://docs.scipy.org/doc/scipy/dev/governance.html)
- [NumPy](https://numpy.org/doc/stable/dev/governance/governance.html#governance)
- [Jupyter](https://jupyter.org/governance/overview.html)
