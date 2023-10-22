# Governance

The purpose of this document is to formalize the governance process used by the `aeon`
project, to clarify how decisions are made and how the various elements of our community
interact. Our goal is to ensure a transparent, democratic, and inclusive decision-making
process that empowers all community members to contribute to the project.

`aeon` is a community-owned and community-run project. Anyone with an interest in the
project can join the community, contribute to the project design and participate in the
decision-making process.

This document establishes the decision-making structure used by `aeon` which strives to
find consensus from the community, while avoiding any deadlocks. Our decision-making
process involves different roles, each with specific responsibilities.

## Roles

### Contributors

Contributors are community members who have contributed in concrete ways to the project.
Anyone can become a contributor, and contributions can take many forms – not only code –
as detailed in the contributing guide. Contributors play a crucial role in shaping the
project through participating in discussions and influencing the decision-making
process.

### Core Developers

Core developers are community members that have made significant contributions and are
trusted to continue the development of the project and engage with its community.
Core developers are granted write access to the `aeon` repository and voting rights
on all project decisions. They are expected to review code contributions
and engage with topics or code they are knowledgeable about.

New core developers are nominated by existing core developers and are subject to a
two-thirds majority vote of existing core developers. Core developers are expected to
maintain a reasonable amount of engagement with the project. Developing code,
interacting with contributions and engaging with the broader community are all valid
contributions for core developers.

Core developers who no longer want or are able to engage with the project are expected
to resign. Core developers may lose their role after a year of inactivity with no
contributions (see above) and engagement with fellow developers. If this lack of
engagement has been demonstrated, another core developer can suggest removal and create
a pull request on the `aeon` repository. The removal will complete if this pull request
is successfully merged.

### Workgroups

Extra responsibilities are delegated to workgroups, which are groups of contributors
with a core developer lead. Workgroups are responsible for specific areas of the project
and are expected to manage and make decisions in their area of responsibility.
Workgroups are expected to be transparent about their activities and decisions, and to
engage with the community where possible.

Membership of workgroups and the leadership position of a workgroup is subject to a
two-thirds majority vote of core developers. Workgroup members unable to fulfill their
responsibilities are expected to resign from the workgroup. It is the responsibility of
the workgroup lead to safeguard access to relevant project resources and accounts.

#### Infrastructure Workgroup

The infrastructure workgroup maintains the infrastructure of `aeon` to ensure the
smooth running of the project. This includes ensuring the website remains online, that
CI remains functional and other related tasks.

The infrastructure workgroup maintains ownership of the `aeon` GitHub organisation and
ReadTheDocs account.

#### Release Management Workgroup

The release management workgroup manages `aeon` releases. This includes deciding on
release schedules, managing release candidates and publishing releases.

The release management workgroup maintains ownership of the `aeon` PyPi project and
conda feedstock.

#### Finance Workgroup

The finance workgroup is responsible for managing the project finances. This includes
approving any project expenses and managing any finance related accounts.

#### Communications Workgroup

The role of the communications workgroups is to interact with the broader community
through the `aeon` social network accounts and discussion forums. It is the
responsibility of the communications workgroup to manage and maintain the `aeon`
Twitter, LinkedIn, Slack and other relevant communications accounts.

The communications team maintains access to social networking accounts and is
responsible for managing access to the `aeon` email address. To help manage
GitHub discussions, the communication workgroup is given triage access to the `aeon`
GitHub repository.

#### Code of Conduct Workgroup

The Code of Conduct Workgroup (CoCW) consists of contributors tasked with making sure
`aeon` remains a welcoming and inclusive community. CoCW responsibilities include
maintaining the `aeon` Code of Conduct (CoC) and managing reports of breaking the CoC.
CoCW members are expected to review reports of CoC violations, contact and discuss with
involved individuals and make recommendations on actions to take.

Any CoCW members involved in a CoC report or CoCW members which have a conflict of
interest regarding the report are expected to recuse themselves from the process. The
CoCW is given triage access to the `aeon` GitHub repository to moderate discussions if
necessary.

## Decision-Making Process

Decisions about the future of the project are announced publicly to allow discussion
with all members of the community. The whole process from proposal to implementation
is fully visible, apart from topics considered sensitive. All non-sensitive project
management discussion takes place on the contributors channel on the project
Slack and/or the issue tracker. Occasionally, sensitive discussion and votes such as
appointments will occur in private Slack channels.

For most decisions a consensus seeking process of all interested contributors is used.
Contributors try to find a resolution that has no open objections among core developers.
If a reasonable amount of time (at least 7-days for non-trivial changes) has passed
since the last change to a proposed contribution, the proposal has at least one approval
(+1) and no rejections (-1) from core developers, it can be approved by lazy consensus.
If a change is rejected, it is expected that an explanation and description of
conditions (if any) to withdraw the rejection is provided.

At any point during the discussion, any core developer in favour of a change can call
for a vote, which will conclude two weeks from the call for the vote. Any vote to
bypass a rejection from a core developer must be backed by an AEP (see the following
section). For major contributions (such as a new module or major framework redesigns)
an AEP may be requested without a rejection or vote. In the event a vote is called,
the proposal must receive a two-thirds majority of core developers to be approved.

All changes to the `aeon` code or documentation should be done via Pull Request.
By default, push rights to the `main` GitHub branch are restricted for all core developers.
In emergencies, the infrastructure workgroup may temporarily revoke the branch
protection for group members and make direct commits, but this should happen only in
emergencies where harm will come to the project unless timely action is taken.

### Enhancement Proposals

For contentious decision-making votes (not including appointment votes), a proposal
must have been made public for discussion before the vote. It is recommended that this
proposal is made as a consolidated document, in the form of an “aeon Enhancement
Proposal” (AEP). The AEP template is available
[here](https://github.com/aeon-toolkit/aeon-admin/blob/main/aep/aep_template.md), but
the use of said template is not a requirement. A detailed issue or pull request can
substitute an AEP if all parties believe it is sufficient, but a more formal proposal
can be requested by any core developer.

Having a rejection on a pull request does not necessitate the creation of an AEP and
further discussion to find consensus can be held, but one must be created prior to any
vote to bypass a rejection.

A vote does not have to be called for an AEP to be submitted. If a contributor wants to
make a more formal proposal or believes a change will be controversial, an AEP can be
submitted to the community for discussion and comment.

## Acknowledgements

Substantial portions of this document were adapted from or inspired by the following
projects governance documents:

- [Scikit-learn](https://scikit-learn.org/stable/governance.html)
- [SciPy](https://docs.scipy.org/doc/scipy/dev/governance.html)
- [NumPy](https://numpy.org/doc/stable/dev/governance/governance.html#governance)
- [Jupyter](https://jupyter.org/governance/overview.html)
- [Pandas](https://pandas.pydata.org/about/governance.html)
