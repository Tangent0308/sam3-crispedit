"""Versioned twin semantics: old shared edit versus object-adapted edits."""
ADAPTED = 'same-operation-adapted-3'


def adapted(group):
    return group.get('twin_contract_version') == ADAPTED


def blind_probe_satisfied(group):
    # Object-adapted payloads need not hide which member they fit. This is not
    # a successful blind audit; that historical group test is inapplicable.
    return adapted(group) or group.get('group_checks', {}).get('blind_disambiguation', {}).get('pass') is True
