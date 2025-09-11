grant connect, ctxapp, dwrole, unlimited tablespace to your_db_user;
grant execute on ctxsys.ctx_ddl to your_db_user;
grant execute on DBMS_CLOUD to your_db_user;
grant execute on DBMS_CLOUD_AI to your_db_user;
grant execute on DBMS_VECTOR to your_db_user;
grant execute on DBMS_VECTOR_CHAIN to your_db_user;
grant execute on DBMS_CLOUD_PIPELINE to your_db_user;
GRANT EXECUTE ON DBMS_RESULT_CACHE TO your_db_user;


BEGIN
    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(
            privilege_list => xs$name_list('connect'),
            principal_name => 'your_db_user',
            principal_type => xs_acl.ptype_db
        )
    );
END;
/