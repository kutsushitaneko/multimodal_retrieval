grant connect, ctxapp, dwrole, unlimited tablespace to labuser;
grant execute on ctxsys.ctx_ddl to labuser;
grant execute on DBMS_CLOUD_AI to labuser;
grant execute on DBMS_VECTOR to labuser;
grant execute on DBMS_VECTOR_CHAIN to labuser;


BEGIN
    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(
            privilege_list => xs$name_list('connect'),
            principal_name => 'labuser',
            principal_type => xs_acl.ptype_db
        )
    );
END;
/